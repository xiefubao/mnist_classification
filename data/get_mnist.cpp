#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdio>
#include <cmath>
#include "opencv2/opencv.hpp"

using namespace cv; 
using namespace std;
const int width = 28;
const int height = 28;
const double e = 2.71828182845904;
const double pi = 3.14159265359;
int ReverseInt(int i){
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void read_Mnist_Label(string filename, vector<int>&labels){
	ifstream file;
	file.open(filename.c_str(), ios::binary);
	if (file.is_open()){
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		for (int i = 0; i < number_of_images; i++){
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			labels.push_back((int)label);
		}	
	}
}

void read_Mnist_Images(string filename, vector<vector<double> >&images){
	ifstream file(filename.c_str(), ios::binary);
	if (file.is_open()){
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		unsigned char label;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);

		cout << "magic number = " << magic_number << endl;
		cout << "number of images = " << number_of_images << endl;
		cout << "rows = " << n_rows << endl;
		cout << "cols = " << n_cols << endl;

		for (int i = 0; i < number_of_images; i++){
			vector<double>tp;
			for (int r = 0; r < n_rows; r++){
				for (int c = 0; c < n_cols; c++){
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					tp.push_back(image);
				}
			}
			images.push_back(tp);
		}
	}
}
vector<vector<double> > train_images;
vector<int> train_labels;

vector<vector<double> > test_images;
vector<int> test_labels;
void read(){
    read_Mnist_Images("train-images.idx3-ubyte", train_images);
    read_Mnist_Label("train-labels.idx1-ubyte", train_labels);
    read_Mnist_Images("test-images.idx3-ubyte", test_images);
    read_Mnist_Label("test-labels.idx1-ubyte", test_labels);
}
void output0(ofstream& file,int i)
{
    if(i >= 10000)
        file << "";
    else if(i > 1000 )
        file << "0";
    else if(i > 100)
        file << "00";
    else if(i > 10)
        file << "000";
    else 
        file << "0000";
}

int main()
{
    read();
    ofstream file;
    
    file.open("train_data.txt");
    for(int i = 0;i < 60000; ++i)
    {
        output0(file,i);
        file << i << " 28 28 " << endl;
        for(int j = 0;j < 28; ++j){
            for(int k = 0;k < 28; ++k){
                file << train_images[i][j * 28 + k] << (k == 27?"":" " );
            }
            file << endl;
        }
    }
    file.close();

    
    file.open("train_labels.txt");
    for(int i = 0;i < 60000; ++i)
        file << train_labels[i] << endl;
    file.close();



    file.open("test_data.txt");
    for(int i = 0;i < 10000; ++i)
    {
        output0(file,i);
        file << i << " 28 28 " << endl;
        for(int j = 0;j < 28; ++j){
            for(int k = 0;k < 28; ++k){
                file << test_images[i][j * 28 + k] << (k == 27?"":" " );
            }
            file << endl;
        }
    }
    file.close();


    
    file.open("test_labels.txt");
    for(int i = 0;i < 10000; ++i)
        file << test_labels[i] << endl;
    file.close();

    return 0;
}