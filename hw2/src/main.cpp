#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <sstream>
#include <cmath>
#include <ctime>

using std::string;
using std::cout;
using std::endl;

#define FEATURE_SIZE 11392
#define TRAIN_SIZE 2177020
#define TEST_SIZE 220245

#define TRAIN_PIECE_SIZE 40000
#define LEARNING_RATE 0.001
#define PARALLEL 55
#define TRAIN_MAX_TIMES 3000

short feature_map[FEATURE_SIZE]; // An map from real features to actual features
int feature_actual;

// Consider that the features appears in test should be important, so make a map.
void get_useful_features() {
  for (int i = 0; i < FEATURE_SIZE; i++)
    feature_map[i] = -1;

  double count = 1;
  int index, feature;
  char c;

  FILE *test_data = fopen("data/test.txt", "r");
  while (fscanf(test_data, "%d", &index) != EOF) {
    while (fscanf(test_data, "%d:1", &feature)) {
      if (feature_map[feature] == -1)
        feature_map[feature] = count++; // map: origin feature -> count
      if ((c = fgetc(test_data)) == '\n')
        break;
    }
  }
  fclose(test_data);
  feature_actual = count + 1;
  cout << "There should be " << feature_actual << " useful features." << endl;
}

// Split train data to 55 pieces
void split_training_data() {
  int pieces = 0;
  int count; // Size of each train data piece
  const string train_piece_prefix = "build/train"; // Prefix for splited train data
  string path, line;

  std::ifstream train_data("data/train.txt");
  while (pieces < PARALLEL) {
    count = TRAIN_PIECE_SIZE;
    std::ostringstream _ss;
    _ss << train_piece_prefix << pieces << ".txt";
    path = _ss.str();

    std::ofstream part_train_data(path);
    while (count-- && (!train_data.eof())) {
      getline(train_data, line);
      part_train_data << line << std::endl;
    }
    part_train_data.close();
    pieces++;
  }
  train_data.close();
  cout << "Data splited." << endl;
}

void read_feature(short piece, bool* list_of_features[], bool reference[]) {
  string path;
  std::ostringstream _ss;
  _ss << "build/train" << piece << ".txt";
  path = _ss.str();

  int feature;
  short ref;
  char c;
  FILE *train_data = fopen(path.c_str(), "r");
  for (int i = 0; i < TRAIN_PIECE_SIZE; i++) {
    fscanf(train_data, "%d", &ref);
    reference[i] = ref; // Get reference
    while (fscanf(train_data, "%d:1", &feature)) {
      if (feature_map[feature] != -1) // If it's the IMPORTANT feature
        list_of_features[i][feature_map[feature]] = 1;
      if ((c = fgetc(train_data)) == '\n')
        break;
    }
  }
  fclose(train_data);
  cout << "Train piece " << piece << " has read." << endl;
}

double sigmoid(double z) {
  return 1 / (1 + std::exp(-z));
}

double cost_function(bool* list_of_features[], double* theta, bool reference[]) {
  double cost = 0, sum, hyphthesis;
  for (int i = 0; i < TRAIN_PIECE_SIZE; i++) {
    sum = 0;
    for (int j = 0; j < feature_actual; j++)
      sum += theta[j] * list_of_features[i][j];
    hyphthesis = sigmoid(sum);
    cost += - reference[i] * std::log(hyphthesis) - (1 - reference[i]) * std::log(1 - hyphthesis);
  }
  return cost;
}

void train(short piece, bool* list_of_features[], double* theta, bool reference[]) {
  double prev_cost = cost_function(list_of_features, theta, reference) + 100; // For entry
  double cost, hyphthesis, difference;
  int count = 0;
  while ((cost = cost_function(list_of_features, theta, reference)) < prev_cost && count++ < TRAIN_MAX_TIMES) {
    prev_cost = cost;
    cout << "Piece " << piece << " current cost: " << cost << endl;
    // Gradient Descent
    for (int i = 0; i < TRAIN_PIECE_SIZE; i++) {
      hyphthesis = 0;
      for (int j = 0; j < feature_actual; j++)
        hyphthesis += theta[j] * list_of_features[i][j];
      difference = sigmoid(hyphthesis) - reference[i];
      for (int j = 0; j < feature_actual; j++) {
        theta[j] -= LEARNING_RATE * difference * list_of_features[i][j];
      }
    }
  }

  // Save theta for each block
  std::ostringstream _ss;
  _ss << "dest/theta" << piece << ".csv";
  std::ofstream theta_data(_ss.str());
  for (int i = 0; i < feature_actual - 1; i++)
    theta_data << theta[i] << " ";
  theta_data << theta[feature_actual - 1];
  theta_data.close();
}

void predict(short piece, double theta[]) {
  bool* predict_feature = new bool[feature_actual];
  predict_feature[0] = 1;
  for (int i = 1; i < feature_actual; i++)
    predict_feature[i] = 0;
  std::ostringstream _ss;
  _ss << "build/predict" << piece << ".csv";

  std::ofstream predict_data(_ss.str());
  FILE *test_data = fopen("data/test.txt", "r");
  int index, feature, predict_result;
  char c;
  while (fscanf(test_data, "%d", &index) != EOF) {
    while (fscanf(test_data, "%d:1", &feature)) {
      predict_feature[feature_map[feature]] = 1;
      if ((c = fgetc(test_data)) == '\n')
        break;
    }
    predict_result = 0;
    for (int i = 0, predict_result = 0; i < feature_actual; i++)
      predict_result += theta[i] * predict_feature[i];
    predict_data << index << "," << (int)std::round(sigmoid(predict_result)) << endl;
  }
  fclose(test_data);
  predict_data.close();

  delete[] predict_feature;
  cout << "Piece " << piece << " has predicted." << endl;
}

// Train and predict for one piece of train
void train_and_predict(short piece) {
  // Declaration
  bool* list_of_features[TRAIN_PIECE_SIZE];
  for (int i = 0; i < TRAIN_PIECE_SIZE; i++) {
    list_of_features[i] = new bool[feature_actual];
    list_of_features[i][0] = true; // The first feature, always be 1
    for (int j = 1; j < feature_actual; j++)
      list_of_features[i][j] = false;
  }

  bool reference[TRAIN_PIECE_SIZE];
  double *theta = new double[feature_actual];
  for (int i = 0; i < feature_actual; i++)
    theta[i] = 0;

  // Read feature
  read_feature(piece, list_of_features, reference);

  // Train
  train(piece, list_of_features, theta, reference);

  for (int i = 0; i < TRAIN_PIECE_SIZE; i++)
    delete[] list_of_features[i];

  // Predict
  predict(piece, theta);
  delete[] theta;
}

// Combine all predicts, and then choose the more reasonable one.
void combine_predict() {
  string predict_piece_prefix = "build/predict";
  bool list_of_predicts[TEST_SIZE];

  int index, predict_result;
  for (int i = 0; i < PARALLEL - 1; i++) {
    std::ostringstream _ss;
    _ss << predict_piece_prefix << i << ".csv";

    FILE *predict_data = fopen(_ss.str().c_str(), "r");
    while (fscanf(predict_data, "%d,%d\n", &index, &predict_result) != EOF)
      list_of_predicts[index] += predict_result;
    fclose(predict_data);
  }

  std::ofstream combine_predict_data("dest/submissions.csv");
  combine_predict_data << "id,label" << endl;

  for (int i = 0; i < TEST_SIZE; i++)
    combine_predict_data << i << "," << (list_of_predicts[i] > (PARALLEL - 1) / 2 ? 1 : 0) << endl;
  combine_predict_data.close();

  cout << "Predict finished. Thank you." << endl;
}

int main(int argc, char const *argv[]) {
  get_useful_features();
  // split_training_data();

  clock_t t = clock();
#pragma omp parallel for
  for (int i = 0; i < PARALLEL - 1; i++) // Ignore The final...
    train_and_predict(i);

  combine_predict();
  t = clock() - t;
  cout << "It used:" << ((float)t)/CLOCKS_PER_SEC << endl;
  return 0;
}