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

MPI_Datatype createMPIBodyType() {
    const int nitems = 5; // Number of elements in Body struct
    int blocklengths[5] = {1, 1, 1, 1, 1}; // All elements are of length 1
    MPI_Datatype types[5] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE}; // All elements are of type MPI_DOUBLE
    MPI_Datatype MPI_BODY_TYPE;
    MPI_Aint offsets[5];

    offsets[0] = offsetof(Body, x);
    offsets[1] = offsetof(Body, y);
    offsets[2] = offsetof(Body, mass);
    offsets[3] = offsetof(Body, vx);
    offsets[4] = offsetof(Body, vy);

    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_BODY_TYPE);
    MPI_Type_commit(&MPI_BODY_TYPE);

    return MPI_BODY_TYPE;
}

std::vector<Body> distributeBodies(const std::vector<Body>& all_bodies, int world_size, int world_rank, MPI_Datatype MPI_BODY_TYPE) {
    int total_bodies = all_bodies.size();
    int bodies_per_process = total_bodies / world_size;
    int remaining_bodies = total_bodies % world_size;

    // Calculate the actual number of bodies for each process
    int local_body_count;
    if (world_rank < remaining_bodies) {
        local_body_count = bodies_per_process + 1;
    } else {
        local_body_count = bodies_per_process;
    }

    std::vector<Body> local_bodies(local_body_count);

    if (world_rank == 0) {
        // Prepare counts and displacements for uneven distribution
        std::vector<int> counts(world_size);
        std::vector<int> displacements(world_size);

        int current_displacement = 0;
        for (int i = 0; i < world_size; ++i) {
            counts[i] = (i < remaining_bodies) ? bodies_per_process + 1 : bodies_per_process;
            displacements[i] = current_displacement;
            current_displacement += counts[i];
        }

        MPI_Scatterv(all_bodies.data(), counts.data(), displacements.data(), MPI_BODY_TYPE, 
                     local_bodies.data(), local_body_count, MPI_BODY_TYPE, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_BODY_TYPE, 
                     local_bodies.data(), local_body_count, MPI_BODY_TYPE, 0, MPI_COMM_WORLD);
    }

    return local_bodies;
}

std::vector<Body> gatherBodies(const std::vector<Body>& local_bodies, int world_size, int world_rank, MPI_Datatype MPI_BODY_TYPE, int total_bodies) {
    std::vector<Body> all_bodies;
    if (world_rank == 0) {
        // Allocate memory for all bodies in the root process
        all_bodies.resize(total_bodies);
    }

    int local_body_count = local_bodies.size();
    std::vector<int> counts(world_size);
    std::vector<int> displacements(world_size);

    // Gather the number of bodies each process will send
    MPI_Gather(&local_body_count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        // Calculate displacements
        int current_displacement = 0;
        for (int i = 0; i < world_size; ++i) {
            displacements[i] = current_displacement;
            current_displacement += counts[i];
        }
    }

    // Gather all bodies to the root process
    MPI_Gatherv(local_bodies.data(), local_body_count, MPI_BODY_TYPE, 
                all_bodies.data(), counts.data(), displacements.data(), MPI_BODY_TYPE, 
                0, MPI_COMM_WORLD);

    return all_bodies;
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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    std::string inputFile, outputFile;
    int steps;
    double theta, dt;
    bool visualization = false;

    // Parse arguments and read input file
    if (world_rank == 0) {
        // Only root process handles file I/O
        parseArguments(argc, argv, inputFile, outputFile, steps, theta, dt, visualization);
    }

    // Broadcast parsed arguments to all processes
    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&theta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::vector<Body> bodies;
    if (world_rank == 0) {
        bodies = readBodiesFromFile(inputFile);
    }

    // Broadcast bodies to all processes
    int num_bodies = bodies.size();
    MPI_Bcast(&num_bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (world_rank != 0) {
        bodies.resize(num_bodies);
    }
    MPI_Bcast(bodies.data(), num_bodies * sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);

    auto start = std::chrono::high_resolution_clock::now();

    QuadTree tree(MAX_SIZE);
    for (const auto& body : bodies) {
        tree.insert(body);
    }


    for (int step = 0; step < steps; ++step) {
        // Root process builds the quadtree
        if (world_rank == 0) {
            for (const auto& body : bodies) {
                tree.insert(body);
            }

        }

        // Serialize and broadcast the quadtree structure
        std::vector<char> serializedTree;
        if (world_rank == 0) {
            serializeQuadTree(tree.root, serializedTree);
        }

        // Broadcast the size of the serialized data
        int serializedSize = serializedTree.size();
        MPI_Bcast(&serializedSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Resize buffer for other processes
        if (world_rank != 0) {
            serializedTree.resize(serializedSize);
        }

        // Broadcast the serialized tree
        MPI_Bcast(serializedTree.data(), serializedSize, MPI_BYTE, 0, MPI_COMM_WORLD);

        // Receive and deserialize quadtree at other processes
        if (world_rank != 0) {
            int position = 0;
            QuadTreeNode* root = deserializeQuadTree(serializedTree, position);
            // Now you have the reconstructed QuadTree on other processes
        }

        // Perform local computation on each process
        for (auto& body : bodies) {
            // Update forces and positions based on the received quadtree
        }

        // Collect updates from all processes at the root
        // TODO: Use MPI_Gather or similar to collect the updated body data at root
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    if (world_rank == 0) {
        std::cout << "Simulation time: " << elapsed.count() << " seconds." << std::endl;
        writeBodiesToFile(outputFile, bodies);
    }

    MPI_Finalize();
    return 0;
}