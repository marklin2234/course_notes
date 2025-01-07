#include <iostream>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 5) {
    std::cerr << "Incorrect number of arguments.\n";
    return 1;
  }
  const char *const serv_addr = argv[1];
  const char *const n_port = argv[2];
  const char *const req_code = argv[3];
  const char *const msg = argv[4];

  int sockfd, status;
  struct addrinfo hints, *servinfo;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_STREAM;

  status = getaddrinfo(serv_addr, n_port, &hints, &servinfo);

  if (status < 0) {
    std::cerr << "ERROR failed to get server address info.\n";
    return 1;
  }

  if ((sockfd = socket(servinfo->ai_family, servinfo->ai_socktype,
                       servinfo->ai_protocol)) < 0) {
    std::cerr << "ERROR creating socket.\n";
    return 1;
  }

  if (connect(sockfd, servinfo->ai_addr, servinfo->ai_addrlen) < 0) {
    std::cerr << "ERROR connecting with server.\n";
    return 1;
  }

  int bytes_sent = 0;
  int num_bytes = sizeof(req_code);

  while (bytes_sent < num_bytes) {
    int n = send(sockfd, req_code + bytes_sent, num_bytes - bytes_sent, 0);

    if (n < 0) {
      std::cerr << "ERROR send failed.\n";
      return 1;
    }
    bytes_sent += n;
  }

  char buffer[256];
  memset(buffer, 0, 256);

  status = recv(sockfd, buffer, 255, 0);

  if (status < 0) {
    std::cerr << "ERROR failed to recieve with errno: " << errno << "\n";
    return 1;
  }
  buffer[status] = '\0';

  std::string r_port{buffer};

  hints.ai_socktype = SOCK_DGRAM;
  status = getaddrinfo(serv_addr, r_port.c_str(), &hints, &servinfo);

  if (status < 0) {
    std::cerr << "ERROR failed to get server address info.\n";
    return 1;
  }

  close(sockfd);
  if ((sockfd = socket(servinfo->ai_family, servinfo->ai_socktype,
                       servinfo->ai_flags)) < 0) {
    std::cout << "ERROR failed to create UDP socket.\n";
    return 1;
  }

  int n = sendto(sockfd, msg, strlen(msg), 0, servinfo->ai_addr,
                 servinfo->ai_addrlen);
  if (n < 0) {
    std::cerr << "ERROR failed to send message.\n";
    return 1;
  }

  status = recvfrom(sockfd, buffer, 255, 0, servinfo->ai_addr,
                    &servinfo->ai_addrlen);

  if (status < 0) {
    std::cerr << "ERROR failed to receive from UDP connection.\n";
    return 1;
  }
  buffer[status] = '\0';

  std::cout << "Returned word: " << std::string{buffer} << "\n";
  close(sockfd);
  return 0;
}
