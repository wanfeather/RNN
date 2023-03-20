#ifndef MATRIX_H
#define MATRIX_H

typedef struct _matrix
{
    int row, col;
    double *element;
}Mat;

Mat *new_matrix(int, int);
void init_matrix(Mat *);
void set_element(Mat *, int, int, double);
void print_matrix(Mat *);
Mat *matrix_product(Mat *, Mat *);
double elementwise(Mat *, Mat *);
Mat *transpose(Mat *);
void delete_matrix(Mat *);
void copy_matrix(Mat *, Mat *);
Mat *matrix_addtion(Mat *, Mat *);
Mat *element_product(Mat *, Mat *);
Mat *scale_product(double, Mat *);

#endif