#include <mpi.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <chrono>
#include <cmath>


#include "BarnesHut.h" // Assume this header includes your QuadTree and Body structures
#include <cstring>

void writeBodiesToFile(const std::string &fileName, const std::vector<Body> &bodies);
//void simulate(std::vector<Body>& bodies, int steps, double theta, double dt);



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
    const int nitems = 8; // Number of elements in Body
    int blocklengths[nitems] = {1, 1, 1, 1, 1, 1, 1, 1}; // All elements are of length 1
    MPI_Datatype types[nitems] = {MPI_INT, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Aint offsets[nitems];

    offsets[0] = offsetof(Body, index);
    offsets[1] = offsetof(Body, x);
    offsets[2] = offsetof(Body, y);
    offsets[3] = offsetof(Body, mass);
    offsets[4] = offsetof(Body, vx);
    offsets[5] = offsetof(Body, vy);
    offsets[6] = offsetof(Body, fx);
    offsets[7] = offsetof(Body, fy);

    MPI_Datatype mpi_body_type;
    MPI_Type_create_struct(nitems, blocklengths, offsets, types, &mpi_body_type);
    MPI_Type_commit(&mpi_body_type);

    return mpi_body_type;
}


std::vector<Body> readBodiesFromFile(const std::string &fileName) {
    // Read the input file and return a vector of Body structures
    std::vector<Body> bodies;
    std::ifstream file(fileName);

    if (!file.is_open()) {
        throw std::runtime_error("Unable to open file: " + fileName);
    }

    int numBodies;
    file >> numBodies;

    for (int i = 0; i < numBodies; ++i) {
        Body body;
        file >> body.index >> body.x >> body.y >> body.mass >> body.vx >> body.vy;
        bodies.push_back(body);
        //std::cout << "Body "  << body.index << " x=" << body.x << " y=" << body.y << std::endl;
    }

    file.close();
    return bodies;
}

/*

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
*/




void insertBody(QuadTreeNode* node, Body* body) {
    if (node->body == nullptr) {
        // If the node is empty, place the body here
        node->body = body;
        return;
    }

    // If the node already contains a body, create children and move the existing body to one of the children
    if (node->NW == nullptr) {
        // Create children
        double halfWidth = node->width / 2.0;
        double x = node->x;
        double y = node->y;
        node->NW = new QuadTreeNode(x - halfWidth / 2, y + halfWidth / 2, halfWidth);
        node->NE = new QuadTreeNode(x + halfWidth / 2, y + halfWidth / 2, halfWidth);
        node->SW = new QuadTreeNode(x - halfWidth / 2, y - halfWidth / 2, halfWidth);
        node->SE = new QuadTreeNode(x + halfWidth / 2, y - halfWidth / 2, halfWidth);
        
        // Insert the existing body into one of the new children
        insertBody(node, node->body);
        node->body = nullptr; // Remove body from the current node
    }

    // Determine in which child node to insert the new body
    if (body->x < node->x) {
        if (body->y > node->y) {
            insertBody(node->NW, body);
        } else {
            insertBody(node->SW, body);
        }
    } else {
        if (body->y > node->y) {
            insertBody(node->NE, body);
        } else {
            insertBody(node->SE, body);
        }
    }

    // Update the mass and center of mass for this node
    node->mass += body->mass;
    node->x = (node->mass * node->x + body->mass * body->x) / (node->mass + body->mass);
    node->y = (node->mass * node->y + body->mass * body->y) / (node->mass + body->mass);
}

void QuadTree::insert(const Body& body) {
    Body* newBody = new Body(body); // Copy the body to store in the tree
    insertBody(root, newBody);
}

void QuadTree::updateForces(Body& body, double theta) {
    if (body.mass == -1) {
        // Particle is lost, skip updating forces
        return;
    }

    body.fx = 0;
    body.fy = 0;
    calculateForce(body, root, theta, body.fx, body.fy);
}

void QuadTree::clear() {
    clearNode(root);
    root = nullptr;
}

void QuadTree::clearNode(QuadTreeNode* node) {
    if (node != nullptr) {
        clearNode(node->NW);
        clearNode(node->NE);
        clearNode(node->SW);
        clearNode(node->SE);
        delete node;
    }
}

void serializeBody(const Body* body, std::vector<char>& buffer) {
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&body->x), reinterpret_cast<const char*>(&body->x) + sizeof(body->x));
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&body->y), reinterpret_cast<const char*>(&body->y) + sizeof(body->y));
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&body->vx), reinterpret_cast<const char*>(&body->vx) + sizeof(body->vx));
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&body->vy), reinterpret_cast<const char*>(&body->vy) + sizeof(body->vy));
    buffer.insert(buffer.end(), reinterpret_cast<const char*>(&body->mass), reinterpret_cast<const char*>(&body->mass) + sizeof(body->mass));
}

Body* deserializeBody(const std::vector<char>& buffer, int& pos) {
    Body* body = new Body();

    std::memcpy(&body->x, &buffer[pos], sizeof(body->x)); pos += sizeof(body->x);
    std::memcpy(&body->y, &buffer[pos], sizeof(body->y)); pos += sizeof(body->y);
    std::memcpy(&body->vx, &buffer[pos], sizeof(body->vx)); pos += sizeof(body->vx);
    std::memcpy(&body->vy, &buffer[pos], sizeof(body->vy)); pos += sizeof(body->vy);
    std::memcpy(&body->mass, &buffer[pos], sizeof(body->mass)); pos += sizeof(body->mass);

    return body;
}


void serializeQuadTree(QuadTreeNode* node, std::vector<char>& buffer) {
    if (node == nullptr) {
        buffer.push_back(0); // Null marker
        return;
    }

    buffer.push_back(1); // Non-null marker
    // Serialize node data
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&node->x), reinterpret_cast<char*>(&node->x) + sizeof(node->x));
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&node->y), reinterpret_cast<char*>(&node->y) + sizeof(node->y));
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&node->mass), reinterpret_cast<char*>(&node->mass) + sizeof(node->mass));
    buffer.insert(buffer.end(), reinterpret_cast<char*>(&node->width), reinterpret_cast<char*>(&node->width) + sizeof(node->width));

    // Serialize the body (if not nullptr)
    if (node->body != nullptr) {
        buffer.push_back(1); // Body exists marker
        // Serialize Body object (assuming Body is serializable)
        serializeBody(node->body, buffer); // You need to implement serializeBody
    } else {
        buffer.push_back(0); // No body marker
    }

    // Recursively serialize children
    serializeQuadTree(node->NW, buffer);
    serializeQuadTree(node->NE, buffer);
    serializeQuadTree(node->SW, buffer);
    serializeQuadTree(node->SE, buffer);
}


QuadTreeNode* deserializeQuadTree(const std::vector<char>& buffer, int& pos) {
    if (pos >= (int) buffer.size() || buffer[pos] == 0) {
        pos++;
        return nullptr;
    }

    pos++; // Skip non-null marker
    // Deserialize node data
    double x, y, mass, width;
    std::memcpy(&x, &buffer[pos], sizeof(x)); pos += sizeof(x);
    std::memcpy(&y, &buffer[pos], sizeof(y)); pos += sizeof(y);
    std::memcpy(&mass, &buffer[pos], sizeof(mass)); pos += sizeof(mass);
    std::memcpy(&width, &buffer[pos], sizeof(width)); pos += sizeof(width);

    // Create a new node
    QuadTreeNode* node = new QuadTreeNode(x, y, width);
    node->mass = mass;

    // Deserialize the body if it exists
    if (buffer[pos++] == 1) {
        node->body = deserializeBody(buffer, pos); // Implement deserializeBody
    }

    // Recursively deserialize children
    node->NW = deserializeQuadTree(buffer, pos);
    node->NE = deserializeQuadTree(buffer, pos);
    node->SW = deserializeQuadTree(buffer, pos);
    node->SE = deserializeQuadTree(buffer, pos);

    return node;
}

void QuadTree::calculateForce(const Body& body, QuadTreeNode* node, double theta, double& fx, double& fy) {
    if (node == nullptr || node->mass == 0 || body.mass == -1) return;

    // Check if the particle is out of bounds and mark it as lost
    if (body.x < 0 || body.x > 4 || body.y < 0 || body.y > 4) {
        const_cast<Body&>(body).mass = -1; // Mark the particle as lost
        return;
    }

    double dx = node->x - body.x;
    double dy = node->y - body.y;
    double distance = sqrt(dx * dx + dy * dy);

    // Avoid self-interaction and handle RLIMIT
    if (distance == 0) return;
    if (distance < RLIMIT) {
        distance = RLIMIT;
    }

    if (node->body != nullptr || (node->width / distance) < theta) {
        // Treat the node as a single point mass
        double F = (G * node->mass * body.mass) / (distance * distance);
        fx += F * dx / distance; // Force in x direction
        fy += F * dy / distance; // Force in y direction
    } else {
        // Node is not far enough, need to consider its children
        calculateForce(body, node->NW, theta, fx, fy);
        calculateForce(body, node->NE, theta, fx, fy);
        calculateForce(body, node->SW, theta, fx, fy);
        calculateForce(body, node->SE, theta, fx, fy);
    }
}

/*
void QuadTree::clearNode(QuadTreeNode* node) {
    if (node != nullptr) {
        // Recursively delete child nodes
        clearNode(node->NW);
        clearNode(node->NE);
        clearNode(node->SW);
        clearNode(node->SE);

        // Delete the body if it's not nullptr
        delete node->body;

        // Now delete the node itself
        delete node;
    }
}

QuadTree::~QuadTree() {
    clearNode(root);
}
*/

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


// Helper function to calculate the start and end indices for the bodies assigned to each process
void calculateBounds(int total_bodies, int world_rank, int world_size, int &start, int &end) {
    int chunk_size = total_bodies / world_size;
    int remainder = total_bodies % world_size;

    if (world_rank < remainder) {
        // The first 'remainder' ranks get 'chunk_size + 1' bodies each
        start = world_rank * (chunk_size + 1);
        end = start + chunk_size;
    } else {
        // The remaining ranks get 'chunk_size' bodies each
        start = world_rank * chunk_size + remainder;
        end = start + (chunk_size - 1);
    }
}

std::vector<Body> distributeBodiesOld(const std::vector<Body> &all_bodies, int world_rank, int world_size) {
    int total_bodies = all_bodies.size();
    int start, end;

    // Broadcast the total number of bodies to all processes
    MPI_Bcast(&total_bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the subset of bodies for this process
    calculateBounds(total_bodies, world_rank, world_size, start, end);

    // Create a vector to store the local bodies
    std::vector<Body> local_bodies;

    // The root process (rank 0) distributes the bodies to all processes
    if (world_rank == 0) {
        // For the root process, simply copy its portion
        local_bodies.assign(all_bodies.begin() + start, all_bodies.begin() + end + 1);

        // Send each other process its portion of the bodies
        for (int rank = 1; rank < world_size; ++rank) {
            calculateBounds(total_bodies, rank, world_size, start, end);
            MPI_Send(&all_bodies[start], end - start + 1, MPI_BODY, rank, 0, MPI_COMM_WORLD);
        }
    } else {
        // Non-root processes receive their portion of the bodies
        local_bodies.resize(end - start + 1);
        MPI_Recv(&local_bodies[0], end - start + 1, MPI_BODY, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    return local_bodies;
}

std::vector<Body> distributeBodies(std::vector<Body> &all_bodies, int world_rank, int world_size) {
    int total_bodies = all_bodies.size();

    // Broadcast the total number of bodies to all processes
    MPI_Bcast(&total_bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Create a vector to store the bodies
    std::vector<Body> local_bodies(total_bodies);

    // If root process, broadcast all bodies to all processes
    if (world_rank == 0) {
        MPI_Bcast(all_bodies.data(), total_bodies, MPI_BODY, 0, MPI_COMM_WORLD);
    } else {
        // All other processes receive the broadcasted bodies
        MPI_Bcast(local_bodies.data(), total_bodies, MPI_BODY, 0, MPI_COMM_WORLD);
    }

    return local_bodies;
}

void simulate(std::vector<Body>& local_bodies, QuadTree& local_tree, int step, double theta, double dt) {
    // Build the local QuadTree with the current set of bodies
    local_tree.clear(); // Clear the existing tree
    for (const auto& body : local_bodies) {
        local_tree.insert(body);
    }

    // Compute the forces on each body
    for (auto& body : local_bodies) {
        local_tree.updateForces(body, theta);
    }

    // Update the positions and velocities of each body
    for (auto& body : local_bodies) {
        // Calculate acceleration
        double ax = body.fx / body.mass;
        double ay = body.fy / body.mass;

        // Update position
        body.x += body.vx * dt + 0.5 * ax * dt * dt;
        body.y += body.vy * dt + 0.5 * ay * dt * dt;

        // Update velocity
        body.vx += ax * dt;
        body.vy += ay * dt;
    }
}


std::vector<Body> gatherAllBodies(const std::vector<Body>& local_bodies, int world_rank, int world_size) {
    int local_count = local_bodies.size();
    std::vector<int> all_counts(world_size);
    std::vector<int> displacements(world_size);

    // Gather the number of bodies each process will send to the root process
    MPI_Gather(&local_count, 1, MPI_INT, all_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Only the root process prepares to receive all bodies
    std::vector<Body> all_bodies;
    if (world_rank == 0) {
        int total_count = 0;
        for (int i = 0; i < world_size; ++i) {
            displacements[i] = total_count;
            total_count += all_counts[i];
        }
        all_bodies.resize(total_count);
    }

    // Gather all bodies to the root process
    MPI_Gatherv(local_bodies.data(), local_count, MPI_BODY, 
                all_bodies.data(), all_counts.data(), displacements.data(), 
                MPI_BODY, 0, MPI_COMM_WORLD);

    return all_bodies;
}

void updateQuadTree(QuadTree& local_tree, const std::vector<Body>& all_bodies) {
    // Clear the existing tree to prepare for new data
    local_tree.clear();

    // Insert all updated bodies into the QuadTree
    for (const auto& body : all_bodies) {
        local_tree.insert(body);
    }
}


std::vector<Body> updateLocalBodies(QuadTree& tree, const std::vector<Body>& local_bodies, double dt, double theta) {
    std::vector<Body> updated_bodies;

    for (const auto& body : local_bodies) {
        // Create a copy of the body to update
        Body updated_body = body;

        // Reset force
        updated_body.fx = 0.0;
        updated_body.fy = 0.0;

        // Calculate force exerted on this body by others
        tree.calculateForce(updated_body, tree.root, theta, updated_body.fx, updated_body.fy);

        //tree.calculateForce(updated_body, theta);

        // Update velocity based on the force
        updated_body.vx += updated_body.fx / updated_body.mass * dt;
        updated_body.vy += updated_body.fy / updated_body.mass * dt;

        // Update position based on the velocity
        updated_body.x += updated_body.vx * dt;
        updated_body.y += updated_body.vy * dt;

        // Add the updated body to the list
        updated_bodies.push_back(updated_body);
    }

    return updated_bodies;
}

std::vector<Body> gatherAndBroadcastUpdates(const std::vector<Body>& local_updates, int world_size) {
    int local_update_count = local_updates.size();
    std::vector<Body> global_updates(local_update_count * world_size);

    std::cout << "local_update_count: " << local_update_count << std::endl;
    std::cout << "global_updates size: " << global_updates.size() << std::endl;

    // Gather updates from all processes and broadcast them to all processes
    MPI_Allgather(local_updates.data(), local_update_count, MPI_BODY, global_updates.data(), local_update_count, MPI_BODY, MPI_COMM_WORLD);

    // print global_updates
    for (int i = 0; i < global_updates.size(); i++) {
        std::cout << "global_updates[" << i << "]: " << global_updates[i].index << std::endl;
    }

    return global_updates;
}

void applyUpdates(std::vector<Body>& local_bodies, const std::vector<Body>& updates) {
    // Assuming the updates vector contains updated bodies in the same order as the local_bodies vector
    for (size_t i = 0; i < local_bodies.size(); ++i) {
        local_bodies[i] = updates[i];
    }
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    MPI_BODY = createMPIBodyType();


    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Command line arguments
    std::string inputFile, outputFile;
    int steps;
    double theta, dt;
    bool visualization = false;

    // Parse arguments and read bodies
    std::vector<Body> all_bodies;
    if (world_rank == 0) {
        parseArguments(argc, argv, inputFile, outputFile, steps, theta, dt, visualization);
        all_bodies = readBodiesFromFile(inputFile);
    }


    // Broadcast the number of steps, theta, and dt to all processes
    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&theta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::cout << world_rank << " | " << steps << " " << theta << " " << dt << std::endl;

    // Broadcast the size of all_bodies to all processes
    int total_bodies = (world_rank == 0) ? all_bodies.size() : 0;
    MPI_Bcast(&total_bodies, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::cout << world_rank << " | total_bodies: " << total_bodies << std::endl;

    // Resize the all_bodies vector for non-root processes
    if (world_rank != 0) {
        all_bodies.resize(total_bodies);
    }

    // Broadcast all bodies to all processes
    MPI_Bcast(all_bodies.data(), total_bodies, MPI_BODY, 0, MPI_COMM_WORLD);


    // Determine local bodies for each process
    int local_start = world_rank * (all_bodies.size() / world_size);
    int local_end = (world_rank + 1) * (all_bodies.size() / world_size);
    std::cout << world_rank << " | local_start: " << local_start << " local_end: " << local_end << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    // Main simulation loop
    for (int step = 0; step < steps; ++step) {
        std::cout << world_rank << " | step: " << step << std::endl;
        // Create and build the quad tree for all bodies
        QuadTree tree(MAX_SIZE); // Define MAX_SIZE appropriately
        for (const auto& body : all_bodies) {
            std::cout << world_rank << " | inserting body " << body.index << std::endl;
            tree.insert(body); // Ensure this method exists
        }
        std::cout << world_rank <<  " | Tree built" << std::endl;

        // Compute updates for local bodies
        std::vector<Body> local_bodies(all_bodies.begin() + local_start, all_bodies.begin() + local_end);
        std::cout << world_rank << " | local_bodies size: " << local_bodies.size() << std::endl;

        std::vector<Body> local_updates = updateLocalBodies(tree, local_bodies, dt, theta);
        std::cout << world_rank << " | local_updates size: " << local_updates.size() << std::endl;
        // print local_updates
        for (int i = 0; i < local_updates.size(); i++) {
            std::cout << world_rank << " | local_updates[" << i << "]: " << local_updates[i].index << std::endl;
        }

        // Gather and broadcast updates from all processes
        std::vector<Body> global_updates = gatherAndBroadcastUpdates(local_updates, world_size);
        std::cout << world_rank << " | global_updates size: " << global_updates.size() << std::endl;
        // print out global_updates content
        for (int i = 0; i < global_updates.size(); i++) {
            std::cout << world_rank << " | global_updates[" << i << "]: " << global_updates[i].index << std::endl;
        }
        
        // Apply updates to all bodies
        applyUpdates(all_bodies, global_updates);
        std::cout << world_rank << " | all_bodies size: " << all_bodies.size() << std::endl;

        tree.clear();

        MPI_Barrier(MPI_COMM_WORLD);

    }

    // Clean up
    MPI_Type_free(&MPI_BODY);
    MPI_Finalize();
    return 0;
}


int mainOld(int argc, char *argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    MPI_BODY = createMPIBodyType();


    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Command line arguments
    std::string inputFile, outputFile;
    int steps;
    double theta, dt;
    bool visualization = false;

    // Parse arguments and read bodies
    std::vector<Body> bodies;
    if (world_rank == 0) {
        parseArguments(argc, argv, inputFile, outputFile, steps, theta, dt, visualization);
        bodies = readBodiesFromFile(inputFile);
    }


    // Broadcast the number of steps, theta, and dt to all processes
    MPI_Bcast(&steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&theta, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    std::cout << steps << " " << theta << " " << dt << std::endl;

    
    // Distribute bodies among processes
    std::vector<Body> local_bodies = distributeBodies(bodies, world_rank, world_size);

    std::cout << "local_bodies size: " << local_bodies.size() << std::endl;

    /*
    // Main simulation loop
    for (int step = 0; step < steps; ++step) {
        // Each process updates its local bodies
        //simulate(local_bodies, 1, theta, dt);

        // Gather and broadcast updates
        //std::vector<Body> global_updates = gatherAndBroadcastUpdates(local_bodies, world_size);

        // Update local Quad Tree with global updates
        //updateQuadTree(global_updates);

        // Synchronize all processes
        MPI_Barrier(MPI_COMM_WORLD);
    }
    

    // Gather all bodies to the root process for output
    std::vector<Body> all_bodies = gatherAllBodies(local_bodies, world_rank, world_size);

    if (world_rank == 0) {
        writeBodiesToFile(outputFile, all_bodies);
    }
    */

    
    MPI_Type_free(&MPI_BODY);


    // Finalize MPI
    MPI_Finalize();
    return 0;
}
