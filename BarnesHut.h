
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

public:
    QuadTreeNode *root;

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

    void clearNode(QuadTreeNode* node);

    void clear();

};

MPI_Datatype MPI_BODY;

