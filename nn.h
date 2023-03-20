#ifndef NN_H
#define NN_H

#include"matrix.h"
#include"activation.h"

void Sigmoid(Mat *);
void Tanh(Mat *);
void Relu(Mat *);
void Softmax(Mat *);

void Sigmoid_gradient(Mat *);
void Tanh_gradient(Mat *);
void Relu_gradient(Mat *);


double Cross_Entropy(Mat *, int);

#endif