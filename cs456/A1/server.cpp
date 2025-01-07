#include <algorithm>
#include <cerrno>
#include <iostream>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <unistd.h>

class TCPServer {
private:
  const int REQ_CODE;
  int sockfd;

  void negotiate(int newsockfd, struct sockaddr_in cli_addr) {
    char buffer[256];
    memset(buffer, 0, 256);

    int status = recv(newsockfd, buffer, 255, 0);
    if (status < 0) {
      std::cerr << "ERROR reading from socket: " << newsockfd << "\n";
      return;
    }

    std::string code{buffer};

    if (stoi(code) != REQ_CODE) {
      std::cerr << "ERROR wrong code.\n";
      return;
    }

    int r_sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if (r_sockfd < 0) {
      std::cerr << "ERROR on creating socket.\n";
      return;
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));

    serv_addr.sin_port = htons(0);
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_family = AF_INET;

    if (bind(r_sockfd, (const struct sockaddr *)&serv_addr, sizeof(serv_addr)) <
        0) {
      std::cerr << "ERROR on binding UDP socket.\n";
      return;
    }

    socklen_t slen = sizeof(serv_addr);
    if (getsockname(r_sockfd, (struct sockaddr *)&serv_addr, &slen) < 0) {
      std::cerr << "ERROR failed to get hostname with errno: " << errno << "\n";
      return;
    }

    unsigned short int r_port = ntohs(serv_addr.sin_port);
    std::string port_str{std::to_string(r_port)};
    int n = send(newsockfd, port_str.c_str(), port_str.size(), 0);
    if (n < 0) {
      std::cerr << "ERROR failed to send r_port.\n";
      return;
    }
    close(newsockfd);

    communicate(r_sockfd);

    close(r_sockfd);
  }

  void communicate(int r_sockfd) {
    char buffer[256];
    memset(buffer, 0, 256);

    struct sockaddr_in from;
    socklen_t fromlen = sizeof(from);
    while (true) {
      int n = recvfrom(r_sockfd, buffer, 255, 0, (struct sockaddr *)&from,
                       &fromlen);

      if (n < 0) {
        std::cerr << "Failed to recieve UDP message.\n";
        return;
      }

      std::string s{buffer};

      std::reverse(s.begin(), s.end());

      n = sendto(r_sockfd, s.c_str(), s.size(), 0, (struct sockaddr *)&from,
                 fromlen);

      if (n < 0) {
        std::cerr << "Failed to send UDP message.\n";
        return;
      }
    }
  }

public:
  TCPServer(int code)
      : REQ_CODE{code}, sockfd{socket(AF_INET, SOCK_STREAM, 0)} {
    if (sockfd < 0) {
      std::cerr << "ERROR on creating socket.\n";
    }

    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = INADDR_ANY;
    serv_addr.sin_port = htons(3000);

    if (bind(sockfd, (const struct sockaddr *)&serv_addr, sizeof(serv_addr)) <
        0) {
      std::cerr << "ERROR on binding socket.\n";
    }

    socklen_t len = sizeof(serv_addr);
    if (getsockname(sockfd, (struct sockaddr *)&serv_addr, &len) < 0) {
      std::cerr << "ERROR on getsockname().\n";
      close(sockfd);
    }
    std::cout << "SERVER_PORT=" << ntohs(serv_addr.sin_port) << "\n";
  }
  ~TCPServer() { close(sockfd); }

  void acceptConnections() {
    int newsockfd, pid;
    socklen_t clilen;
    struct sockaddr_in cli_addr;

    listen(sockfd, 5);
    while (true) {
      newsockfd = accept(sockfd, (struct sockaddr *)&cli_addr, &clilen);

      if (newsockfd < 0) {
        std::cerr << "ERROR on accept.\n";
      }

      pid = fork();

      if (pid < 0) {
        std::cerr << "ERROR on fork.\n";
      } else if (pid == 0) {
        close(sockfd);
        negotiate(newsockfd, cli_addr);
        return;
      } else {
        close(newsockfd);
      }
    }
  }
};

int main(int argc, char **argv) {
  if (argc == 1) {
    std::cerr << "Not enough arguments.\n";
    return 0;
  }
  int req_code = std::stoi(argv[1]);

  TCPServer server{req_code};

  server.acceptConnections();
}
