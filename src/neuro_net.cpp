#include "neuro_net.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <fstream>

/*ОПИСАНИЕ ОДНОГО НЕЙРОНА*/
    Neuro::Neuro(double value, double error):
        value(value),
        active_value(1),
        error(error)
    {}
    void Neuro::sigma(bool is_derive){
        double buf = exp(-value);
        if(is_derive)
            error *= active_value * (1 - active_value);
        else
            active_value = 1 / (1 + buf);
    }

/*ОПИСАНИЕ СЕТИ*/
    Net::Net(std::vector<size_t>& nneuros_in, double etta, double moment):
        nlayers(nneuros_in.size()),
        etta(etta),
        moment(moment),
        nneuros(nneuros_in)
    {
        /*инициализация сети слоями + нейронами*/

        /*плюс нейрон для свободного члена*/
        for(size_t layer = 0; layer < nlayers - 1; ++layer)
            nneuros[layer]++;

        /*заполнение связей и нейронов*/
        for(size_t layer = 0; layer < nlayers - 2; ++layer){
            std::vector<Neuro> layer_fill(nneuros[layer] - 1);
            layer_fill.push_back(1);
            neuros.push_back(layer_fill);

            std::vector<std::vector<double>> link;
            std::vector<std::vector<double>> delta_link;
            for(size_t line = 0; line < nneuros[layer]; ++line){
                std::vector<double> line_fill;
                for(size_t elem = 0; elem < nneuros[layer + 1] - 1; ++elem){
                    double fix = 1.0 * (rand() % 10000) / 10000 - 0.5;
                    line_fill.push_back(fix);
                }
                std::vector<double> delta_fill(nneuros[layer + 1] - 1, 0);
                link.push_back(line_fill);
                delta_link.push_back(delta_fill);
            }
            links.push_back(link);
            delta_links.push_back(delta_link);
        }

        /*особенности со свободным членом*/
        std::vector<Neuro> layer_fill(nneuros[nlayers - 2] - 1);
        layer_fill.push_back(1);
        neuros.push_back(layer_fill);
        std::vector<Neuro> buf(nneuros[nlayers - 1]);
        neuros.push_back(buf);

        std::vector<std::vector<double>> link;
        std::vector<std::vector<double>> delta_link;
        for(size_t line = 0; line < nneuros[nlayers - 2]; ++line){
            std::vector<double> line_fill;
            for(size_t elem = 0; elem < nneuros[nlayers - 1]; ++elem){
                double fix = 1.0 * (rand() % 10000) / 10000 - 0.5;
                line_fill.push_back(fix);
            }
            std::vector<double> delta_fill(nneuros[nlayers - 1], 0);
            link.push_back(line_fill);
            delta_link.push_back(delta_fill);
        }
        links.push_back(link);
        delta_links.push_back(delta_link);
        find_max_layer();
    }

    Net::Net(char const* infile_name, double etta, double moment):
        etta(etta),
        moment(moment)
    {
        std::ifstream in(infile_name, std::ios::in);
        if(!in.is_open()){
            std::cout << "Can't open infile" << std::endl;
        } else {
            in >> nlayers;
            for(size_t matrix = 0; matrix < nlayers; ++matrix){
                size_t n_cur, n_next;
                in >> n_cur >> n_next;
                nneuros.push_back(n_cur);
                std::vector<std::vector<double>> matrix_fill;
                std::vector<std::vector<double>> deltamatrix_fill;
                for(size_t line = 0; line < n_cur; ++line){
                    std::vector<double> line_fill;
                    for(size_t elem = 0; elem < n_next; ++elem){
                        double buf_elem;
                        in >> buf_elem;
                        line_fill.push_back(buf_elem);
                    }
                    deltamatrix_fill.push_back(std::vector<double>(n_next, 0));
                    matrix_fill.push_back(line_fill);
                }
                delta_links.push_back(deltamatrix_fill);
                links.push_back(matrix_fill);
            }
            nlayers++;
            nneuros.push_back(links[nlayers - 2][0].size());
            for(auto nneuro : nneuros){
                std::vector<Neuro> neuros_fill(nneuro);
                neuros.push_back(neuros_fill);
            }
        }
    }

    void Net::set_parametres(double education, double momentum){
        etta = education;
        moment = momentum;
    }

    void Net::print_neuros() const{
        /*вывод нейронов сети*/

        std::cout << "Data: Neurons X Nlayers(val, er):" << std::endl;
        std::cout.precision(2);
        std::cout.setf(std::ios::fixed);
        for(size_t neuro = 0; neuro < Max_layer_size; ++neuro){
            for(size_t layer = 0; layer < nlayers; ++layer){
                if(nneuros[layer] > neuro){
                    std::cout << std::setw(4) << std::right <<'('
                         << std::setw(6) << std::left << neuros[layer][neuro].active_value
                         << ','
                         << std::setw(6) << std::left << neuros[layer][neuro].error
                         << ')';
                } else {
                    std::cout << std::setw(18) << ' ';
                }
            }
            std::cout << std::endl;
        }
        std::cout << "\n\n";
        std::cout.unsetf(std::ios::fixed);
    }

    void Net::print_links() const{
        /*вывод синапсов сети*/

        std::cout.precision(2);
        for(size_t layer = 0; layer < nlayers - 2; ++layer){
            std::cout << "Links between layer_" << layer + 1 << " and layer_" << layer + 2 << std::endl;
            for(size_t lines = 0; lines < nneuros[layer]; ++lines){
                for(size_t columns = 0; columns < nneuros[layer + 1] - 1; ++columns){
                    std::cout << links[layer][lines][columns] << '\t';
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout << "Links between layer_" << nlayers - 1 << " and layer_" << nlayers << std::endl;
        for(size_t lines = 0; lines < nneuros[nlayers - 2]; ++lines){
            for(size_t columns = 0; columns < nneuros[nlayers - 1]; ++columns){
                std::cout << links[nlayers - 2][lines][columns] << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << "\n\n";
    }

    bool Net::download_data(std::vector<double>& in){
        /*загружаем входной слой*/

        if(in.size() + 1 != nneuros[0])
            return false;
        for(size_t neuro = 0; neuro < in.size(); ++neuro){
            neuros[0][neuro].active_value = in[neuro];
        }
        return true;
    }

    void Net::forward_pass(){
        /*прямой проход по сети*/

        for(size_t layer = 0; layer < nlayers - 1; ++layer){
            for(size_t new_neuro = 0; new_neuro < nneuros[layer + 1] - 1; ++new_neuro){
                neuros[layer + 1][new_neuro].value = 0;
                for(size_t old_neuro = 0; old_neuro < nneuros[layer]; ++old_neuro){
                    neuros[layer + 1][new_neuro].value +=
                            neuros[layer][old_neuro].active_value * links[layer][old_neuro][new_neuro];
                }
                neuros[layer + 1][new_neuro].sigma();
            }
        }
        for(size_t new_neuro = 0; new_neuro < nneuros[nlayers - 1]; ++new_neuro){
            neuros[nlayers - 1][new_neuro].value = 0;
            for(size_t old_neuro = 0; old_neuro < nneuros[nlayers - 2]; ++old_neuro){
                neuros[nlayers - 1][new_neuro].value +=
                        neuros[nlayers - 2][old_neuro].active_value * links[nlayers - 2][old_neuro][new_neuro];
            }
            neuros[nlayers - 1][new_neuro].sigma();
        }
    }

    std::vector<double> Net::get_result(){
        /*выгружаем выходной слой*/

        std::vector<double> result;
        for(auto Neuro : neuros[nlayers - 1]){
            result.push_back(Neuro.active_value);
        }
        return result;
    }

    void Net::back_propagation(std::vector<double> result){
        /*обучам выходным вектором*/

        if(result.size() != neuros[nlayers - 1].size()){
            std::cout << "Error::back_propagation" << std::endl;
            return;
        }

        /*ищем дельты*/
        for(size_t neuron = 0; neuron < nneuros[nlayers - 1]; ++neuron){
            neuros[nlayers - 1][neuron].error = result[neuron] - neuros[nlayers - 1][neuron].active_value;
            neuros[nlayers - 1][neuron].sigma(true);
        }
        for(size_t neuro = 0; neuro < nneuros[nlayers - 2] - 1; ++neuro){
            neuros[nlayers - 2][neuro].error = 0;
            for(size_t errneuro = 0; errneuro < nneuros[nlayers - 1]; ++errneuro){
                neuros[nlayers - 2][neuro].error +=
                        neuros[nlayers - 1][errneuro].error * links[nlayers - 2][neuro][errneuro];
            }
            neuros[nlayers - 2][neuro].sigma(true);
        }
        for(size_t layer = nlayers - 3; layer > 0; --layer){
            for(size_t neuro = 0; neuro < nneuros[layer] - 1; ++neuro){
                neuros[layer][neuro].error = 0;
                for(size_t errneuro = 0; errneuro < nneuros[layer + 1] - 1; ++errneuro){
                    neuros[layer][neuro].error +=
                            neuros[layer + 1][errneuro].error * links[layer][neuro][errneuro];
                }
                neuros[layer][neuro].sigma(true);
            }
        }
        /*нашли все дельты*/

        /*меняем веса*/
        for(size_t layer = 0; layer < nlayers - 2; ++layer){
            for(size_t prev_neuro = 0; prev_neuro < nneuros[layer]; ++prev_neuro){
                for(size_t neuro = 0; neuro < nneuros[layer + 1] - 1; ++neuro){
                    delta_links[layer][prev_neuro][neuro] *= moment;
                    delta_links[layer][prev_neuro][neuro] +=
                            neuros[layer][prev_neuro].active_value * neuros[layer + 1][neuro].error * etta;
                    links[layer][prev_neuro][neuro] += delta_links[layer][prev_neuro][neuro];
                }
            }
        }
        for(size_t prev_neuro = 0; prev_neuro < nneuros[nlayers - 2]; ++prev_neuro){
            for(size_t neuro = 0; neuro < nneuros[nlayers - 1]; ++neuro){
                delta_links[nlayers - 2][prev_neuro][neuro] *= moment;
                delta_links[nlayers - 2][prev_neuro][neuro] +=
                        neuros[nlayers - 2][prev_neuro].active_value * neuros[nlayers - 1][neuro].error * etta;
                links[nlayers - 2][prev_neuro][neuro] += delta_links[nlayers - 2][prev_neuro][neuro];
            }
        }
        /*изменили все веса*/
    }

    void Net::dump_links(char const* outfile_name){
        std::ofstream out(outfile_name, std::ios::out);
        if(!out.is_open()){
            std::cerr << "Can't open file for dump" << std::endl;
            return;
        }
        out << nlayers - 1 << std::endl;
        for(auto matrix : links){
            out << matrix.size() << ' ' << matrix[0].size() << std::endl;
            for(auto line : matrix){
                for(auto elem : line){
                    out << elem << ' ';
                }
                out << std::endl;
            }
        }
        out.close();
    }

    void Net::find_max_layer(){ //вычисляет размер наибольшего слоя
        Max_layer_size = 0;
        for(auto layer_size : nneuros){
            if(layer_size > Max_layer_size)
                Max_layer_size = layer_size;
        }
    }
