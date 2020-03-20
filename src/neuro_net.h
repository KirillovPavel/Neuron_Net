#ifndef NEURO_NET_H
#define NEURO_NET_H

#include <vector>

struct Neuro{
    double value;
    double active_value;
    double error;
    Neuro(double value = 0, double error = 0);
    void sigma(bool is_derive = false);
};

struct Net{
/* Main information */
    double etta; //скорость обучения
    double moment; // инерция
    size_t nlayers;
    std::vector<size_t> nneuros;
    std::vector<std::vector<Neuro>> neuros;
    std::vector<std::vector<std::vector<double>>> links;
    std::vector<std::vector<std::vector<double>>> delta_links;

/*Information for optimization*/
    size_t Max_layer_size;

    Net(std::vector<size_t>& nneuros_in, double etta = 0.7, double moment = 0.5);
    Net(char const* infile_name, double etta = 0.7, double moment = 0.5);
    void set_parametres(double education = 0.7, double momentum = 0.5);
    void print_neuros() const;
    void print_links() const;
    bool download_data(std::vector<double>& in);
    void forward_pass();
    std::vector<double> get_result();
    void back_propagation(std::vector<double> result);
    void dump_links(char const* outfile_name);
    void find_max_layer();
};

#endif // NEURO_NET_H
