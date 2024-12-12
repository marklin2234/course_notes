#include <fstream>
#include <iostream>
#include <string>

#define max(x, y) ((x > y) ? x : y)

using namespace std;

bool inDict(string &s, ifstream &file) {
  file.clear();
  file.seekg(0);
  string in;

  while (getline(file, in)) {
    if (in == s) {
      return true;
    }
  }

  return false;
}

int numDiff(string &s1, string &s2) {
  int n = s1.size();

  int cnt = 0;
  for (int i = 0; i < n; i++) {
    if (s1[i] != s2[i])
      cnt++;
  }
  return cnt;
}

bool isSuboptimal(string &s) {
  ifstream file("usedWords.txt");
  string prev, curr;
  getline(file, prev);

  while (getline(file, curr)) {
    if (numDiff(s, prev) == 1) {
      file.close();
      return true;
    }
    prev = curr;
  }
  file.close();
  return false;
}

int main(int argc, char **argv) {
  string start = argv[1];
  string end = argv[2];
  ifstream file(argv[3]);
  ofstream usedWords("usedWords.txt");
  usedWords << start << "\n";
  usedWords.close();

  string in;

  if (!inDict(start, file) || !inDict(end, file)) {
    cerr << "Error: Starting or ending word not found in words file\n";
    usedWords.close();
    return 1;
  }

  cout << "Starting Word: " << start << "\n";
  string prev = start;
  int turn = 1;
  int bestScore = 0;

  while (getline(cin >> ws, in)) {
    if (numDiff(in, prev) != 1) {
      std::cerr << "Error: " << in << " does not differ from " << prev
                << " by exactly one character\n";
      continue;
    } else if (!inDict(in, file)) {
      std::cerr << "Error: " << in << " does not belong to word file\n";
      continue;
    }
    if (isSuboptimal(in)) {
      std::cerr << "This word could have been played earlier\n";
    }
    turn++;
    prev = in;
    usedWords.open("usedWords.txt", ios_base::app);
    usedWords << in << "\n";
    usedWords.close();
    if (in == end) {
      int score = 8 - turn + 1;
      bestScore = max(bestScore, score);
      std::cout << "Congratulations! Your Score: " << score << "\n";
      std::cout << "Best Score: " << bestScore << "\n";
      std::cout << "Starting Word: " << start << "\n";
      turn = 0;
      prev = start;
      usedWords.open("usedWords.txt");
      usedWords << start << "\n";
      usedWords.close();
    }
    if (turn == 8) {
      std::cout << "You lose\n";
      usedWords.close();
      return 0;
    }
    usedWords.close();
  }
}
