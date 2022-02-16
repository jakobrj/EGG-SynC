#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

// https://www.cse.cuhk.edu.hk/~taoyf/course/infs4205/lec/rtree.pdf

class RTree_Node {
public:
    int dims;
    RTree_Node *parent = nullptr;
    std::vector<std::pair < int, float *>*>
    points;
    std::vector<RTree_Node *> children;
    float *mbr_max;
    float *mbr_min;

    RTree_Node(int dims, int B) {
        this->dims = dims;
        this->mbr_max = new float[dims];
        this->mbr_min = new float[dims];
        points.reserve(B + 1);
        children.reserve(B + 1);
    }

    ~RTree_Node() {
        delete mbr_max;
        delete mbr_min;
    }

    int number_of_entries() {
        return children.size() == 0 ? points.size() : children.size();
    }

    void force_mbr() {
        for (auto child: this->children) {
            child->force_mbr();
        }
        recompute_MBR();
    }

    void print_all_points() {
        for (auto child: this->children) {
            child->print_all_points();
        }
        if (this->is_leaf()) {
            for (auto p: points) {
                printf(" %d", p->first);
            }
            printf("\n");
        }
    }

    bool check() {
        if (is_leaf()) {
            for (auto p: this->points) {
                for (int i = 0; i < dims; i++) {
                    if (p->second[i] > this->mbr_max[i] || p->second[i] < this->mbr_min[i]) {
                        printf("leaf\n");
                        return false;
                    }
                }
            }
        } else {
            for (auto child: this->children) {
                for (int i = 0; i < dims; i++) {
                    if (child->parent != this) {
                        printf("internal - bad parent reference\n");
                        return false;
                    }

                    if (child->mbr_max[i] > this->mbr_max[i] || child->mbr_min[i] < this->mbr_min[i]) {
                        printf("internal\n");
                        if (child->is_leaf()) {
                            printf("child is leaf\n");
                        }
                        if (this->is_root()) {
                            printf("this is root\n");
                        }

                        printf("child max: ");
                        for (int i = 0; i < dims; i++) {
                            printf(", %f", child->mbr_max[i]);
                        }
                        printf("\n");
                        printf("child min: ");
                        for (int i = 0; i < dims; i++) {
                            printf(", %f", child->mbr_min[i]);
                        }
                        printf("\n");

                        printf("this max: ");
                        for (int i = 0; i < dims; i++) {
                            printf(", %f", this->mbr_max[i]);
                        }
                        printf("\n");
                        printf("this min: ");
                        for (int i = 0; i < dims; i++) {
                            printf(", %f", this->mbr_min[i]);
                        }
                        printf("\n");
                        return false;
                    }
                    if (!child->check()) {
                        printf("above internal\n");
                        return false;
                    }
                }
            }
        }

        return true;
    }

    bool is_leaf() {
        return this->children.size() == 0;
    }

    bool is_root() {
        return this->parent == nullptr;
    }

    void remove(std::pair<int, float *> *p) {
        for (int i = 0; i < this->points.size(); i++) {
            std::pair<int, float *> *q = this->points[i];
            if (p->first == q->first) {
                this->points.erase(this->points.begin() + i);
                delete q;
                return;
            }
        }
    }


    bool contains(std::pair<int, float *> *p) {
        for (int i = 0; i < this->points.size(); i++) {
            std::pair<int, float *> *q = this->points[i];
            if (p->first == q->first) {
                return true;
            }
        }
        return false;
    }

    bool subtree_contains(std::pair<int, float *> *p) {
        bool contained = false;
        for (auto child: children) {
            contained = contained || child->subtree_contains(p);
        }
        return contained || contains(p);
    }

    void remove(RTree_Node *u) {
        for (int i = 0; i < this->children.size(); i++) {
            RTree_Node *v = this->children[i];
            if (u == v) {
                this->children.erase(this->children.begin() + i);
                return;
            }
        }
    }

    void recompute_MBR() {
        if (is_leaf()) {
            for (int j = 0; j < this->points.size(); j++) {
                std::pair<int, float *> *p = this->points[j];
                for (int i = 0; i < this->dims; i++) {
                    if (j == 0) {
                        this->mbr_max[i] = p->second[i];
                        this->mbr_min[i] = p->second[i];
                    } else {
                        if (p->second[i] > this->mbr_max[i]) {
                            this->mbr_max[i] = p->second[i];
                        }
                        if (p->second[i] < this->mbr_min[i]) {
                            this->mbr_min[i] = p->second[i];
                        }
                    }
                }
            }
        } else {
            for (int j = 0; j < this->children.size(); j++) {
                RTree_Node *u = this->children[j];
                for (int i = 0; i < this->dims; i++) {
                    if (j == 0) {
                        this->mbr_max[i] = u->mbr_max[i];
                        this->mbr_min[i] = u->mbr_min[i];
                    } else {
                        if (u->mbr_max[i] > this->mbr_max[i]) {
                            this->mbr_max[i] = u->mbr_max[i];
                        }
                        if (u->mbr_min[i] < this->mbr_min[i]) {
                            this->mbr_min[i] = u->mbr_min[i];
                        }
                    }
                }
            }
        }
    }

    void update_MBR(std::pair<int, float *> *p) {
        for (int i = 0; i < this->dims; i++) {
            if (this->children.size() == 0 && this->points.size() == 0) {
                this->mbr_max[i] = p->second[i];
                this->mbr_min[i] = p->second[i];
            } else {
                if (p->second[i] > this->mbr_max[i]) {
                    this->mbr_max[i] = p->second[i];
                }
                if (p->second[i] < this->mbr_min[i]) {
                    this->mbr_min[i] = p->second[i];
                }
            }
        }
    }

    void update_MBR(RTree_Node *u) {
        for (int i = 0; i < this->dims; i++) {
            if (this->children.size() == 0) {
                this->mbr_max[i] = u->mbr_max[i];
                this->mbr_min[i] = u->mbr_min[i];
            } else {
                if (u->mbr_max[i] > this->mbr_max[i]) {
                    this->mbr_max[i] = u->mbr_max[i];
                }
                if (u->mbr_min[i] < this->mbr_min[i]) {
                    this->mbr_min[i] = u->mbr_min[i];
                }
            }
        }
    }

    void add_child(RTree_Node *u) {
        update_MBR(u);

        this->children.push_back(u);
        u->parent = this;
    }

    void print_mbr() {
        printf("max:");
        for (int i = 0; i < dims; i++) {
            printf(" %f", this->mbr_max[i]);
        }
        printf("\n");
        printf("min:");
        for (int i = 0; i < dims; i++) {
            printf(" %f", this->mbr_min[i]);
        }
        printf("\n");
    }
};

std::vector<int> argsort(const std::vector<std::pair < int, float *> *

> &array,
int i
) {
std::vector<int> indices(array.size());
std::iota(indices
.

begin(), indices

.

end(),

0);
std::sort(indices
.

begin(), indices

.

end(),

[&array, i](
int left,
int right
) -> bool {
// sort indices according to corresponding array element
return array[left]->second[i] < array[right]->second[i];
});

return
indices;
}

std::vector<int> argsort_min(const std::vector<RTree_Node *> &array, int i) {
    std::vector<int> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array, i](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left]->mbr_min[i] < array[right]->mbr_min[i];
              });

    return indices;
}

std::vector<int> argsort_max(const std::vector<RTree_Node *> &array, int i) {
    std::vector<int> indices(array.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
              [&array, i](int left, int right) -> bool {
                  // sort indices according to corresponding array element
                  return array[left]->mbr_max[i] < array[right]->mbr_max[i];
              });

    return indices;
}

void print(std::pair<int, float *> *p, int d) {
    printf("p: %d:", p->first);
    for (int i = 0; i < d; i++) {
        printf(" %f", p->second[i]);
    }
    printf("\n");
}

class RTree {
private:
    int dims;
    int B;
    RTree_Node *root;

//    void insert_2(RTree_Node *u, std::pair<int, float *> *p) {
////        if(p->first == 2596){
////            printf("down:\n");
////            u->print_mbr();
////        }
//
//
//        if (u->is_leaf()) {
//            u->points.push_back(p);
//            u->update_MBR(p);
//            if (u->points.size() > this->B) {//u overflows
//                this->handle_overflow(u);
//            }
//        } else {
//            RTree_Node *v = this->choose_subtree(u, p);
//            this->insert_2(v, p);
//            u->update_MBR(p);
//        }
//
//
////        if(p->first == 2596){
////            printf("up:\n");
////            u->print_mbr();
////        }
//    }

    void insert(RTree_Node *u, std::pair<int, float *> *p) {
        if (u->is_leaf()) {
            u->points.push_back(p);
            u->update_MBR(p);

            if (u->points.size() > this->B) {//u overflows
                this->handle_overflow(u);
            }
        } else {
            u->update_MBR(p);
            RTree_Node *v = this->choose_subtree(u, p);
            this->insert(v, p);
//            u->update_MBR(p);
        }
    }

    void insert(RTree_Node *E) {//todo not sure this is correct
        //when using insert it is assumed that E is a node and that E had a parent that was removed and was not the root.
        //implying that the following show always exists
        RTree_Node *grad_parent = E->parent->parent;//todo not sure this is correct
        RTree_Node *v = this->choose_subtree(grad_parent, E);
        v->add_child(E);
        if (v->children.size() > this->B) {//v overflows
            this->handle_overflow(v);
        }
        while (v != nullptr) {
            v->update_MBR(E);
            v = v->parent;
        }

    }

    RTree_Node *choose_subtree(RTree_Node *u, RTree_Node *E) {
        RTree_Node *best_v;
        float best_inc_perimeter = std::numeric_limits<float>::max();
        for (RTree_Node *v: u->children) {
            float inc_perimeter = 0.;
            for (int j = 0; j < this->dims; j++) {
                if (E->mbr_max[j] > v->mbr_max[j]) {
                    inc_perimeter += E->mbr_max[j] - v->mbr_max[j];
                }
                if (E->mbr_min[j] < v->mbr_min[j]) {
                    inc_perimeter += v->mbr_min[j] - E->mbr_min[j];
                }
            }
            if (inc_perimeter < best_inc_perimeter) {
                best_inc_perimeter = inc_perimeter;
                best_v = v;
            }
        }

        return best_v;
    }

    RTree_Node *choose_subtree(RTree_Node *u, std::pair<int, float *> *p) {
        RTree_Node *best_v;
        float best_inc_perimeter = std::numeric_limits<float>::max();
        for (RTree_Node *v: u->children) {
            float inc_perimeter = 0.;
            for (int j = 0; j < this->dims; j++) {
                if (p->second[j] > v->mbr_max[j]) {
                    inc_perimeter += p->second[j] - v->mbr_max[j];
                } else if (p->second[j] < v->mbr_min[j]) {
                    inc_perimeter += v->mbr_min[j] - p->second[j];
                }
            }
            if (inc_perimeter < best_inc_perimeter) {
                best_inc_perimeter = inc_perimeter;
                best_v = v;
            }
        }

        return best_v;
    }

    void handle_overflow(RTree_Node *u) {


        RTree_Node *u_ = u->is_leaf() ? this->split_leaf(u) : this->split_internal(u);
        if (u->is_root()) {
            this->root = new RTree_Node(this->dims, this->B);
            this->root->add_child(u);
            this->root->add_child(u_);
        } else {
            RTree_Node *w = u->parent;
            w->update_MBR(u);
            w->add_child(u_);
            if (w->children.size() > this->B) {
                handle_overflow(w);
            }

        }
    }


    RTree_Node *split_leaf(RTree_Node *u) {
//        printf("split_leaf\n");
        int m = u->points.size();

        float best_perimeter_sum = std::numeric_limits<float>::max();
        int best_j;
        int best_i;
        float *mbr_max = new float[this->dims];
        float *mbr_min = new float[this->dims];

        for (int j = 0; j < this->dims; j++) {
            std::vector<int> idx = argsort(u->points, j);
            for (int i = ceil(0.4 * this->B) + 1; i < m - ceil(0.4 * this->B); i++) {
                //todo this is bad but easy to optimize
                float perimeter_sum = 0.;

                for (int d = 0; d < this->dims; d++) {
                    mbr_max[d] = u->points[idx[0]]->second[d];
                    mbr_min[d] = u->points[idx[0]]->second[d];
                }

                for (int l = 1; l <= i; l++) {
                    for (int d = 0; d < this->dims; d++) {
                        if (u->points[idx[l]]->second[d] > mbr_max[d])
                            mbr_max[d] = u->points[idx[l]]->second[d];
                        if (u->points[idx[l]]->second[d] < mbr_min[d])
                            mbr_min[d] = u->points[idx[l]]->second[d];
                    }
                }

                for (int d = 0; d < this->dims; d++) {
                    perimeter_sum += mbr_max[d] - mbr_min[d];
                }

                for (int d = 0; d < this->dims; d++) {
                    mbr_max[d] = u->points[idx[i + 1]]->second[d];
                    mbr_min[d] = u->points[idx[i + 1]]->second[d];
                }

                for (int l = i + 2; l < m; l++) {
                    for (int d = 0; d < this->dims; d++) {
                        if (u->points[idx[l]]->second[d] > mbr_max[d])
                            mbr_max[d] = u->points[idx[l]]->second[d];
                        if (u->points[idx[l]]->second[d] < mbr_min[d])
                            mbr_min[d] = u->points[idx[l]]->second[d];
                    }
                }

                for (int d = 0; d < this->dims; d++) {
                    perimeter_sum += mbr_max[d] - mbr_min[d];
                }

                if (perimeter_sum < best_perimeter_sum) {
                    best_perimeter_sum = perimeter_sum;
                    best_j = j;
                    best_i = i;
                }
            }
        }

        std::vector < std::pair < int, float * > * > *u_points = new std::vector < std::pair < int, float * > * >();
        std::vector < std::pair < int, float * > * > *u__points = new std::vector < std::pair < int, float * > * >();
        std::vector<int> idx = argsort(u->points, best_j);
        for (int i = 0; i < m; i++) {
            if (i <= best_i) {
                u_points->push_back(u->points[idx[i]]);
            } else {
                u__points->push_back(u->points[idx[i]]);
            }
        }

        u->points.clear();
        u->points.insert(u->points.begin(), u_points->begin(), u_points->end());

        RTree_Node *u_ = new RTree_Node(this->dims, this->B);
        u_->points.insert(u_->points.begin(), u__points->begin(), u__points->end());

        u->recompute_MBR();
        u_->recompute_MBR();

        delete mbr_max;
        delete mbr_min;
        delete u_points;
        delete u__points;

        return u_;
    }


    RTree_Node *split_internal(RTree_Node *u) {
//        printf("split_internal\n");
        int m = u->children.size();

        float best_perimeter_sum = std::numeric_limits<float>::max();
        int best_j;
        int best_i;
        int best_argsort;
        float *mbr_max = new float[this->dims];
        float *mbr_min = new float[this->dims];
        std::vector<int> idx;

        for (int argsort_f = 0; argsort_f < 2; argsort_f++) {
            for (int j = 0; j < this->dims; j++) {
                if (argsort_f == 0)
                    idx = argsort_min(u->children, j);
                else
                    idx = argsort_max(u->children, j);
                for (int i = ceil(0.4 * this->B) + 1; i < m - ceil(0.4 * this->B); i++) {
                    //todo this is bad but easy to optimize
                    float perimeter_sum = 0.;

                    for (int d = 0; d < this->dims; d++) {
                        mbr_max[d] = u->children[idx[0]]->mbr_max[d];
                        mbr_min[d] = u->children[idx[0]]->mbr_min[d];
                    }

                    for (int l = 1; l <= i; l++) {
                        for (int d = 0; d < this->dims; d++) {
                            if (u->children[idx[l]]->mbr_max[d] > mbr_max[d])
                                mbr_max[d] = u->children[idx[l]]->mbr_max[d];
                            if (u->children[idx[l]]->mbr_min[d] < mbr_min[d])
                                mbr_min[d] = u->children[idx[l]]->mbr_min[d];
                        }
                    }

                    for (int d = 0; d < this->dims; d++) {
                        perimeter_sum += mbr_max[d] - mbr_min[d];
                    }

                    for (int d = 0; d < this->dims; d++) {
                        mbr_max[d] = u->children[idx[i + 1]]->mbr_max[d];
                        mbr_min[d] = u->children[idx[i + 1]]->mbr_min[d];
                    }

                    for (int l = i + 2; l < m; l++) {
                        for (int d = 0; d < this->dims; d++) {
                            if (u->children[idx[l]]->mbr_max[d] > mbr_max[d])
                                mbr_max[d] = u->children[idx[l]]->mbr_max[d];
                            if (u->children[idx[l]]->mbr_min[d] < mbr_min[d])
                                mbr_min[d] = u->children[idx[l]]->mbr_min[d];
                        }
                    }

                    for (int d = 0; d < this->dims; d++) {
                        perimeter_sum += mbr_max[d] - mbr_min[d];
                    }

                    if (perimeter_sum < best_perimeter_sum) {
                        best_perimeter_sum = perimeter_sum;
                        best_j = j;
                        best_i = i;
                        best_argsort = argsort_f;
                    }
                }
            }
        }

        std::vector < RTree_Node * > u_children;
        std::vector < RTree_Node * > u__children;
        if (best_argsort == 0)
            idx = argsort_min(u->children, best_j);
        else
            idx = argsort_max(u->children, best_j);
        for (int i = 0; i < m; i++) {
            if (i <= best_i) {
                u_children.push_back(u->children[idx[i]]);
            } else {
                u__children.push_back(u->children[idx[i]]);
            }
        }
//        u->children = u_children;
        u->children.clear();
        for (RTree_Node *child: u_children) {
            u->add_child(child);
        }
//        u->children.insert(u->children.begin(), u_children.begin(), u_children.end());

        RTree_Node *u_ = new RTree_Node(this->dims, this->B);
        for (RTree_Node *child: u__children) {
            u_->add_child(child);
        }
//        u_->children = u__children;
//        u_->children.insert(u_->children.begin(), u__children.begin(), u__children.end());

//        u->recompute_MBR();
//        u_->recompute_MBR();

        return u_;
    }

    float dist(float *p, float *q) {
        float sum = 0.;
        for (int j = 0; j < this->dims; j++) {
            float sub = p[j] - q[j];
            sum += sub * sub;
        }
        return sqrt(sum);
    }

    void range(std::vector<int> &R, RTree_Node *u, float eps, float *p) {
        if (u->is_leaf()) {
            for (std::pair<int, float *> *pr: u->points) {
                int idx = pr->first;
                float *q = pr->second;
                if (this->dist(p, q) <= eps) {
                    R.push_back(idx);
                }
            }
        } else {
            for (RTree_Node *v: u->children) {

                bool with_in_MBR = true;
                for (int d = 0; d < dims; d++) {
                    with_in_MBR = with_in_MBR && p[d] + eps >= v->mbr_min[d] && p[d] - eps <= v->mbr_max[d];
                }

                if (with_in_MBR) {
                    range(R, v, eps, p);
                }
            }
        }
    }

    RTree_Node *node_query(RTree_Node *u, std::pair<int, float *> *p) {
        if (u->is_leaf()) {
            if (u->contains(p)) {
                return u;
            }
        }

        for (RTree_Node *v: u->children) {
            bool with_in_MBR = true;
            for (int d = 0; d < this->dims; d++) {
                with_in_MBR =
                        with_in_MBR && p->second[d] >= v->mbr_min[d] && p->second[d] <= v->mbr_max[d];
            }
            if (with_in_MBR) {
                RTree_Node *R = node_query(v, p);
                if (R != nullptr) {
                    return R;
                }
            }
        }
        return nullptr;
    }

    RTree_Node *node_query_check_all(RTree_Node *u, std::pair<int, float *> *p) {
        if (u->is_leaf()) {
            if (u->contains(p)) {
                return u;
            }
        }

        for (RTree_Node *v: u->children) {
            RTree_Node *R = node_query_check_all(v, p);
            if (R != nullptr) {
                return R;
            }
        }
        return nullptr;
    }

    //L is a leaf node
    void condense_tree(RTree_Node *L) {
        int m = 0.4 * B;

        RTree_Node *X = L;
        std::vector < RTree_Node * > N;
        while (!X->is_root()) {
            RTree_Node *parent_X = X->parent;
            if (X->number_of_entries() < m) {
                parent_X->remove(X);
                N.push_back(X);
            } else {//if X has not been removed
                X->recompute_MBR();
            }
            X = parent_X;
        }
        //reinsert all the entries of nodes in N
        for (RTree_Node *X: N) {
            for (RTree_Node *child: X->children) {
                insert(child);
            }
            for (std::pair<int, float *> *p: X->points) {
                insert(root, p);
            }
            delete X;
        }
    }

public:

    RTree(int dims, int B) {
        this->dims = dims;
        this->B = B;
        RTree_Node *root = new RTree_Node(this->dims, this->B);
        this->root = root;
    }

    void force_mbr() {
        root->force_mbr();
    }

    bool check() {
        if (root->check()) {
            return true;
        } else {
            printf("root:\n");
            root->print_mbr();
            return false;
        }
    }

    void insert(std::pair<int, float *> *p) {
        this->insert(root, p);

//        RTree_Node *L = root;
//        while (!L->is_leaf()) {
//            L = choose_subtree(L, p);
//        }
//        if (L->points.size() < B) {
//            L->points.push_back(p);
//            //update to root;
//            RTree_Node *v = L;
//            while (v != nullptr) {
//                v->update_MBR(p);
//                v = v->parent;
//            }
//        } else {
//            L->points.push_back(p);
//            handle_overflow(L);
//            RTree_Node *v = L;
//            while (v != nullptr) {
//                v->recompute_MBR();
//                v = v->parent;
//            }
////            RTree_Node *L1 = L;
////            RTree_Node *L2 = split_leaf(L);
////            L->parent->add_child(L2);
////            //update and overflow to root
////            RTree_Node *v = L->parent;
////            while (v != nullptr) {
////                if(v->children.size()>B){
////                    RTree_Node *v1 = v;
////                    RTree_Node *v2 = split_internal(v);
////                    if(v->is_root()){
////                        this->root = new RTree_Node(dims);
////                        this->root->add_child(v1);
////                    }
////                    v->parent->add_child(v2);
////                }
//////                v->update_MBR(L1);
//////                V->update_MBR(L2);
////                v->update_MBR(p);
////                v = v->parent;
////            }
//        }

    }

//    void insert(at::Tensor X) {
//        for (int i = 0; i < X.size(0); i++) {
//            float *p = X[i].data_ptr<float>();
//            this->insert(root, std::make_pair(i, p));
//        }
//    }

    // https://books.google.dk/books?hl=en&lr=&id=i4ri1sWRPEcC&oi=fnd&pg=PR7&ots=3an4OtpG8o&sig=wMUzLDEQR2jEeLMOOw7tZOUEerE&redir_esc=y#v=onepage&q&f=falsehttps://books.google.dk/books?hl=en&lr=&id=i4ri1sWRPEcC&oi=fnd&pg=PR7&ots=3an4OtpG8o&sig=wMUzLDEQR2jEeLMOOw7tZOUEerE&redir_esc=y#v=onepage&q&f=false
    // R-Trees: Theory and Applications p. 11, 12,
    void remove(std::pair<int, float *> *p) {
        RTree_Node *L;
        if (root->is_leaf()) {
            //search all entried of RN and find/remove p
            root->remove(p);
            L = root;
        } else {
            //find all entries of node covering P
            //follow the corresponding subtrees until the leaf L that contains E
            L = node_query(root, p);
            if (L == nullptr) {

                printf("p: %d\n", p->first);
                if (root->subtree_contains(p)) {
                    printf("is in tree\n");

                    L = node_query_check_all(root, p);
                    while (L != nullptr) {
                        printf("L:\n");
                        L->print_mbr();
                        L = L->parent;
                    }
                    print(p, dims);
                }

                throw std::invalid_argument("could not find point to delete");
            }
            //Remove E from L
            L->remove(p);
        }
//        delete p;
//
//        while (L != nullptr) {
//            L->recompute_MBR();
//            L = L->parent;
//        }
//
//        return;
//        printf("condense_tree\n");
        condense_tree(L);
        //if the root has only one child and is not a leaf
        if (!root->is_leaf() && root->children.size() == 1) {
            RTree_Node *new_root = root->children[0];
            delete root;
            root = new_root;
            new_root->parent = nullptr;
        }
        delete p;
    }

    std::vector<int> range(float eps, float *p) {

        std::vector<int> R;
        this->range(R, this->root, eps, p);
        return R;
    }
};