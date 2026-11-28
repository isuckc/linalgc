#include <stddef.h>
#include <stdio.h>

struct Matrix {
  size_t rows, cols;
  float *values;
};

// #define rows(matrix) \
//   ((size_t *)((char *)(matrix) + (offsetof(struct Matrix, rows))))
// #define cols(matrix) \
//   ((size_t *)((char *)(matrix)) + (offsetof(struct Matrix, cols)))

static float mat_at(const struct Matrix *mat, const size_t row,
                    const size_t col) {
  if (!mat)
    return 0.0f;

  size_t pos = row * col;
  if (pos >= mat->rows * mat->cols)
    return 0.0f;

  return mat->values[pos];
}

static void print_mat(const struct Matrix *mat) {
  for (size_t i = 0; i < mat->rows; i++) {
    for (size_t j = 0; j < mat->cols; j++) {
      if (j == 0)
        printf("[ ");

      printf("%f", mat_at(mat, i, j));

      if (j == mat->cols - 1) {
        printf(" ]");
        if (i == mat->rows - 1) {
          printf("\n");
          continue;
        }
        printf(",\n");
        continue;
      }

      printf(", ");
    }
  }
}

int main(void) {

  // size_t *rows = rows(&mat);
  // size_t *cols = rows(&mat);

  // printf("the rows are %zu and the cols are %zu\n", *rows, *cols);

  struct Matrix mat = {.rows = 3,
                       .cols = 3,
                       .values = (float[]){1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f,
                                           0.0f, 0.0f, 1.0f}};

  print_mat(&mat);

  return 0;
}
