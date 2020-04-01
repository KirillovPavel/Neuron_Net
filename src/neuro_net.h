#ifndef NEURO_NET_H
#define NEURO_NET_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>
#include <string>

struct Neuron{
    double value;
    double active_value;
    double error;
    char function;

    Neuron(char function = 0, double value = 0);
    void set_function(std::string const function);
    void sigma(bool is_derive = false);
    void tangens(bool is_derive = false);
    void ReLu(bool is_derive = false);
    void active(bool is_derive = false);
};

struct Net{
/* Main information */
    double etta; //скорость обучения
    double moment; // инерция
    size_t NLayers;
    std::vector<size_t> NNeurons;
    std::vector<std::vector<Neuron>> Neurons;
    std::vector<std::vector<std::vector<double>>> Links;
    std::vector<std::vector<std::vector<double>>> delta_Links;

/*Information for optimize*/
    size_t Max_layer_size;

    Net(std::vector<size_t>& NNeurons_in, double etta = 1, double moment = 0);
    Net(char const* infile_name, double etta = 1, double moment = 0);
    void set_learning_rate(double etta = 0.7);
    void set_inertion(double moment = 0.5);
    bool download_data(std::vector<double>& in);
    void forward_pass();
    void back_propagation(std::vector<double> result);
    std::vector<double> get_result();

    void dump_Links(char const* outfile_name);
    void print_Neurons() const;
    void print_Links() const;
    void find_max_layer();
};

#endif // NEURO_NET_H
