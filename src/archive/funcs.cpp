#include <pybind11/pybind11.h>
#include <array>

namespace py = pybind11;

int add(int i, int j){
    return i+j;
}

struct OctreeNode {
    struct Vec3 {
        float x, y, z;
        Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    };

    Vec3 corner1;
    Vec3 corner2;
    int level;
    std::array<int, 3> index;

    OctreeNode* children[8];

    OctreeNode(Vec3 c1, Vec3 c2, int i, int j, int k, int lvl) : corner1(c1), corner2(c2), index({i, j, k}), level(lvl) {
        // Initialize all children as nullptr
        for (int i = 0; i < 8; ++i) {
            children[i] = nullptr;
        }
    }

    // Destructor
    ~OctreeNode() {
        for (int i = 0; i < 8; ++i) {
            delete children[i];
        }
    }

    void createChildren() {
        // Check if children already exist
        for (int i = 0; i < 8; ++i) {
            if (children[i] != nullptr) {
                return; // Children already created
            }
        }

        // Calculate the midpoints of the current node's bounds
        float midX = (corner1.x + corner2.x) / 2.0f;
        float midY = (corner1.y + corner2.y) / 2.0f;
        float midZ = (corner1.z + corner2.z) / 2.0f;

        // Compute indices for the children and create them
        for (int i = 0; i < 2; ++i) {        // Left to Right (x-axis)
            for (int j = 0; j < 2; ++j) {    // Bottom to Top (y-axis)
                for (int k = 0; k < 2; ++k) { // Front to Back (z-axis)
                    int childI = 2 * index[0] + i;
                    int childJ = 2 * index[1] + j;
                    int childK = 2 * index[2] + k;

                    Vec3 childCorner1 = {
                        i == 0 ? corner1.x : midX,
                        j == 0 ? corner1.y : midY,
                        k == 0 ? corner1.z : midZ
                    };
                    Vec3 childCorner2 = {
                        i == 0 ? midX : corner2.x,
                        j == 0 ? midY : corner2.y,
                        k == 0 ? midZ : corner2.z
                    };

                    children[i + 2 * j + 4 * k] = new OctreeNode(childCorner1, childCorner2, level + 1, childI, childJ, childK);
                }
            }
        }
    }   



    bool isLeaf() const {
        for (int i = 0; i < 8; ++i) {
            if (children[i] != nullptr) {
                return false;
            }
        }
        return true;
    }


};

void findLeafNodes(const OctreeNode* node, std::vector<const OctreeNode*>& leaves) {
    if (node == nullptr) {
        return; // Base case for null node
    }

    if (node->isLeaf()) {
        leaves.push_back(node); // If node is a leaf, add it to the list
    } else {
        // If node is not a leaf, recursively find leaf nodes in each of its children
        for (int i = 0; i < 8; ++i) {
            findLeafNodes(node->children[i], leaves);
        }
    }
}

class Octree {
private:
    std::vector<std::vector<std::vector<OctreeNode>>> grid;
    float xdim, ydim, zdim;
    int num_x, num_y, num_z;

public:
    Octree(float xdim, float ydim, float zdim, int num_x, int num_y, int num_z)
        : xdim(xdim), ydim(ydim), zdim(zdim), num_x(num_x), num_y(num_y), num_z(num_z) {
        initializeGrid();
    }

    void initializeGrid() {
        grid.resize(num_x, std::vector<std::vector<OctreeNode>>(num_y, std::vector<OctreeNode>(num_z)));

        float dx = xdim / num_x;
        float dy = ydim / num_y;
        float dz = zdim / num_z;

        for (int i = 0; i < num_x; ++i) {
            for (int j = 0; j < num_y; ++j) {
                for (int k = 0; k < num_z; ++k) {
                    OctreeNode::Vec3 c1(i * dx, j * dy, k * dz);
                    OctreeNode::Vec3 c2((i + 1) * dx, (j + 1) * dy, (k + 1) * dz);
                    grid[i][j][k] = OctreeNode(c1, c2, 0);
                }
            }
        }
    }

};



PYBIND11_MODULE(funcs, m) {
    m.doc() = "THB-spline domain octree datastructure";
    m.def("add", &add, "example function");
}