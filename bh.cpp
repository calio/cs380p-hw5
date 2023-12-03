#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <chrono>


const double G = 0.0001; // Gravitational constant
const double MAX_SIZE = 4.0;
const double DT = 0.005;
const double RLIMIT = 0.03;

struct Body {
    int index;
    double x, y;   // Position
    double mass;
    double vx, vy; // Velocity
    double fx, fy; // Forces
};

struct QuadTreeNode {
    double x, y; // Center of mass
    double mass;
    double width; // Width of the region
    Body *body; // Pointer to a body, nullptr for internal nodes
    QuadTreeNode *NW, *NE, *SW, *SE; // Children

    QuadTreeNode(double x, double y, double width)
        : x(x), y(y), mass(0), width(width), body(nullptr), NW(nullptr), NE(nullptr), SW(nullptr), SE(nullptr) {}
};


class QuadTree {
    QuadTreeNode *root;

public:
    QuadTree(double width) {
        root = new QuadTreeNode(0, 0, width);
        // Initialize the tree
    }

    ~QuadTree() {
        delete root;
    }

    void insert(const Body& body);

    void calculateForce(const Body& body, QuadTreeNode* node, double theta, double& fx, double& fy);

    void updateForces(Body& body, double theta);

};

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
    body.fx = 0;
    body.fy = 0;
    calculateForce(body, root, theta, body.fx, body.fy);
}

void QuadTree::calculateForce(const Body& body, QuadTreeNode* node, double theta, double& fx, double& fy) {
    if (node == nullptr || node->mass == 0) return;

    double dx = node->x - body.x;
    double dy = node->y - body.y;
    double distance = sqrt(dx * dx + dy * dy);

    // Avoid self-interaction
    if (distance == 0) return;
    if (distance < RLIMIT) {
        distance = RLIMIT;
    }

    // Check if the node is sufficiently far away or is a leaf node
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



void simulate(std::vector<Body>& bodies, int steps, double theta, double dt) {
    for (int step = 0; step < steps; ++step) {
        QuadTree tree(MAX_SIZE); 
        for (const auto& body : bodies) {
            tree.insert(body);
        }

        for (auto& body : bodies) {
            tree.updateForces(body, theta);

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
}
void writeBodiesToFile(const std::string &fileName, const std::vector<Body> &bodies) {
    std::ofstream outFile(fileName);
    
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open file " << fileName << " for writing.\n";
        return;
    }

    // Write the number of bodies first
    outFile << bodies.size() << std::endl;

    // Write each body's data
    for (const auto& body : bodies) {
        outFile << body.index << " "
                << body.x << " "
                << body.y << " "
                << body.mass << " "
                << body.vx << " "
                << body.vy << std::endl;
    }

    outFile.close();
}

int main(int argc, char *argv[]) {
    std::string inputFile, outputFile;
    int steps;
    double theta, dt;
    bool visualization = false;

    dt = DT;
    parseArguments(argc, argv, inputFile, outputFile, steps, theta, dt, visualization);

    std::vector<Body> bodies = readBodiesFromFile(inputFile);

    auto start = std::chrono::high_resolution_clock::now();

    simulate(bodies, steps, theta, dt);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << elapsed.count() << std::endl;


    writeBodiesToFile(outputFile, bodies);

    return 0;
}
