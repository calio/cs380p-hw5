#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>

#include "BarnesHut.h" // Assume this header includes your QuadTree and Body structures
#include <cstring>

std::vector<Body> readBodiesFromFile(const std::string &fileName);
void writeBodiesToFile(const std::string &fileName, const std::vector<Body> &bodies);
void simulate(std::vector<Body>& bodies, int steps, double theta, double dt);

void parseArguments(int argc, char *argv[], std::string &inputFile, std::string &outputFile, int &steps, double &theta, double &dt, bool &visualization) {
    if (argc < 6) {
        std::cerr << "Not enough arguments. Usage: <program> -i <inputfile> -o <outputfile> -s <steps> -t <theta> -d <dt> [-V]" << std::endl;
        exit(EXIT_FAILURE);
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i") {
            inputFile = argv[++i];
        } else if (arg == "-o") {
            outputFile = argv[++i];
        } else if (arg == "-s") {
            steps = std::stoi(argv[++i]);
        } else if (arg == "-t") {
            theta = std::stod(argv[++i]);
        } else if (arg == "-d") {
            dt = std::stod(argv[++i]);
        } else if (arg == "-V") {
            visualization = true;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            exit(EXIT_FAILURE);       
        }
    }
}

std::vector<Body> distributeBodies(const std::vector<Body>& all_bodies, int rank, int size) {
    int total_bodies = all_bodies.size();
    int bodies_per_process = total_bodies / size;
    int remaining_bodies = total_bodies % size;

    // Calculate the number of bodies for each process and the starting index for each chunk
    std::vector<int> counts(size, bodies_per_process);
    std::vector<int> displs(size, 0);

    for (int i = 0; i < remaining_bodies; ++i) {
        counts[i]++;
    }

    int sum = 0;
    for (int i = 0; i < size; ++i) {
        displs[i] = sum;
        sum += counts[i];
    }

    std::vector<Body> local_bodies(counts[rank]);

    MPI_Scatterv(all_bodies.data(), counts.data(), displs.data(), MPI_BYTE, 
                 local_bodies.data(), counts[rank] * sizeof(Body), MPI_BYTE, 
                 0, MPI_COMM_WORLD);

    return local_bodies;
}


void serializeQuadTree(QuadTreeNode* node, std::vector<char>& buffer) {
    if (node == nullptr) {
        // Use a special marker for null pointers
        buffer.push_back(0);
        return;
    }

    // Serialize node data
    buffer.push_back(1); // Non-null marker
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&node->x), reinterpret_cast<char*>(&node->x) + sizeof(node->x));
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&node->y), reinterpret_cast<char*>(&node->y) + sizeof(node->y));
    // ... Serialize other node attributes similarly ...

    // Recursively serialize children
    serializeQuadTree(node->NW, buffer);
    serializeQuadTree(node->NE, buffer);
    serializeQuadTree(node->SW, buffer);
    serializeQuadTree(node->SE, buffer);
}

QuadTreeNode* deserializeQuadTree(const std::vector<char>& buffer, int& pos) {
    if (pos >= buffer.size() || buffer[pos] == 0) {
        pos++;
        return nullptr;
    }

    pos++; // Skip the non-null marker

    // Deserialize node data
    double x, y;
    std::memcpy(&x, &buffer[pos], sizeof(x));
    pos += sizeof(x);
    std::memcpy(&y, &buffer[pos], sizeof(y));
    pos += sizeof(y);
    // ... Deserialize other node attributes similarly ...

    // Create a new node
    QuadTreeNode* node = new QuadTreeNode(x, y, /* other attributes */);

    // Recursively deserialize children
    node->NW = deserializeQuadTree(buffer, pos);
    node->NE = deserializeQuadTree(buffer, pos);
    node->SW = deserializeQuadTree(buffer, pos);
    node->SE = deserializeQuadTree(buffer, pos);

    return node;
}

QuadTreeNode* deserializeQuadTree(const std::vector<char>& buffer, int& pos) {
    if (pos >= buffer.size() || buffer[pos] == 0) {
        pos++;
        return nullptr;
    }

    pos++; // Skip the non-null marker

    // Deserialize node data
    double x, y;
    std::memcpy(&x, &buffer[pos], sizeof(x));
    pos += sizeof(x);
    std::memcpy(&y, &buffer[pos], sizeof(y));
    pos += sizeof(y);
    // ... Deserialize other node attributes similarly ...

    // Create a new node
    QuadTreeNode* node = new QuadTreeNode(x, y, /* other attributes */);

    // Recursively deserialize children
    node->NW = deserializeQuadTree(buffer, pos);
    node->NE = deserializeQuadTree(buffer, pos);
    node->SW = deserializeQuadTree(buffer, pos);
    node->SE = deserializeQuadTree(buffer, pos);

    return node;
}

void updateBody(Body& body, double dt) {
    // Calculate acceleration
    double ax = body.fx / body.mass;
    double ay = body.fy / body.mass;

    // Update velocity
    body.vx += ax * dt;
    body.vy += ay * dt;

    // Update position
    body.x += body.vx * dt + 0.5 * ax * dt * dt;
    body.y += body.vy * dt + 0.5 * ay * dt * dt;
}

void gatherBodies(std::vector<Body>& all_bodies, const std::vector<Body>& local_bodies, int world_rank, int world_size) {
    int local_count = local_bodies.size();
    std::vector<int> counts(world_size);
    std::vector<int> displs(world_size);

    // Gather the number of bodies each process is sending
    MPI_Gather(&local_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // Calculate displacements and resize the all_bodies vector on the root process
        int total_count = 0;
        for (int i = 0; i < world_size; ++i) {
            displs[i] = total_count;
            total_count += counts[i];
        }
        all_bodies.resize(total_count);
    }

    // Gather the bodies from all processes
    MPI_Gatherv(local_bodies.data(), local_count * sizeof(Body), MPI_BYTE,
                all_bodies.data(), counts.data(), displs.data(), MPI_BYTE,
                0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::string inputFile, outputFile;
    int steps;
    double theta, dt;
    bool visualization = false;

    // Only rank 0 process should read the input file and distribute the bodies
    std::vector<Body> bodies;
    if (world_rank == 0) {
        // Parse arguments and read input file
        parseArguments(argc, argv, inputFile, outputFile, steps, theta, dt, visualization);
        bodies = readBodiesFromFile(inputFile);
    }

    // Distribute the bodies among MPI processes
    std::vector<Body> local_bodies = distributeBodies(bodies, world_rank, world_size);

    // Begin timing the simulation
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before starting timing
    double start_time = MPI_Wtime();

    std::vector<Body> all_updated_bodies;

    // Perform simulation steps
    for (int step = 0; step < steps; ++step) {
        // Each process creates its local tree and computes forces
        QuadTree local_tree(MAX_SIZE);
        for (const auto& body : local_bodies) {
            local_tree.insert(body);
        }

        for (auto& body : local_bodies) {
            local_tree.updateForces(body, theta);
            updateBody(body, dt);  // Update body position and velocity
        }

        // Gather updated bodies from all processes to process 0
        gatherBodies(all_updated_bodies, local_bodies, world_rank, world_size);
        // Synchronize all processes here if necessary
        MPI_Barrier(MPI_COMM_WORLD);

        // Placeholder for MPI gathering operation
    }

    // End timing the simulation
    MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before ending timing
    double end_time = MPI_Wtime();

    if (world_rank == 0) {
        // Process 0 writes results to file
        writeBodiesToFile(outputFile, all_updated_bodies);
        // Print elapsed time
        std::cout << "Simulation time: " << (end_time - start_time) << " seconds." << std::endl;
    }

    // Finalize MPI
    MPI_Finalize();
    return 0;
}
