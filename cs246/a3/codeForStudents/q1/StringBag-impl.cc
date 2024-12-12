module stringBag;
import <iostream>;
import <string>;
import <utility>;

using namespace std;

// Node workhorse ctor
StringBag::Node::Node(const string &s, size_t arity, Node *next)
    : s{s}, arity{arity}, next{next} {}
// StringBag default ctor
StringBag::StringBag() : numElements{0}, numValues{0}, first{nullptr} {}
// StringBag copy ctor
StringBag::StringBag(const StringBag &otherSB)
    : numElements{otherSB.numElements}, numValues{otherSB.numValues} {
  Node *o_curr = otherSB.first;
  Node *curr = new Node(o_curr->s, o_curr->arity, nullptr);
  this->first = curr;

  while (o_curr->next) {
    o_curr = o_curr->next;
    Node *tmp = new Node(o_curr->s, o_curr->arity, nullptr);
    curr->next = tmp;
    curr = curr->next;
  }
}
// StringBag move ctor
StringBag::StringBag(StringBag &&otherSB)
    : numElements{otherSB.numElements}, numValues{otherSB.numValues},
      first{otherSB.first} {
  otherSB.numElements = 0;
  otherSB.numValues = 0;
  otherSB.first = nullptr;
}
// StringBag dtor
StringBag::~StringBag() {
  Node *curr = this->first;
  while (curr) {
    Node *tmp = curr->next;
    delete curr;
    curr = tmp;
  }
}

// StringBag copy operator=
StringBag &StringBag::operator=(const StringBag &otherSB) {
  if (this == &otherSB) {
    return *this;
  }
  StringBag tmp{otherSB};
  std::swap(*this, tmp);
  return *this;
}
// StringBag move operator=
StringBag &StringBag::operator=(StringBag &&otherSB) {
  if (this == &otherSB) {
    return *this;
  }
  this->numElements = otherSB.getNumElements();
  this->numValues = otherSB.getNumValues();
  this->first = otherSB.first;

  otherSB.numElements = 0;
  otherSB.numValues = 0;
  otherSB.first = nullptr;

  return *this;
}

void StringBag::add(const string &s) {
  this->numElements++;

  Node *node;
  if ((node = this->find(s))) {
    node->arity++;
  } else {
    node = new Node(s, 1, nullptr);
    node->next = this->first;
    this->first = node;
    this->numValues++;
  }
}
void StringBag::remove(const string &s) {
  Node *node = this->find(s);

  if (!node || node->arity == 0) {
    return;
  }

  this->numElements--;
  node->arity--;

  if (node->arity == 0) {
    this->numValues--;
  }
}
void StringBag::removeAll(const string &s) {
  Node *node = this->find(s);

  if (!node || node->arity == 0) {
    return;
  }

  this->numElements -= node->arity;
  node->arity = 0;
  this->numValues--;
}

StringBag StringBag::operator+(const StringBag &otherSB) const {
  StringBag ret{*this};

  Node *curr = otherSB.first;

  while (curr) {
    Node *node = ret.find(curr->s);
    ret.numElements += curr->arity;

    if (node) {
      node->arity += curr->arity;
    } else {
      Node *newNode = new Node(curr->s, curr->arity, nullptr);
      newNode->next = ret.first;
      ret.first = newNode;
      ret.numValues++;
    }

    curr = curr->next;
  }

  return ret;
}
StringBag StringBag::operator-(const StringBag &otherSB) const {
  StringBag ret(*this);

  Node *curr = otherSB.first;

  while (curr) {
    Node *node = ret.find(curr->s);
    if (!node)
      continue;

    ret.numElements -= min(node->arity, curr->arity);
    node->arity = max((size_t)0, node->arity - curr->arity);
    if (node->arity == 0) {
      ret.numValues--;
    }
    curr = curr->next;
  }

  return ret;
}
StringBag &StringBag::operator+=(const StringBag &otherSB) {
  Node *curr = otherSB.first;

  while (curr) {
    Node *node = this->find(curr->s);
    this->numElements += curr->arity;

    if (node) {
      node->arity += curr->arity;
    } else {
      Node *newNode = new Node(curr->s, curr->arity, nullptr);
      newNode->next = this->first;
      this->first = newNode;
      this->numValues++;
    }

    curr = curr->next;
  }
  return *this;
}
StringBag &StringBag::operator-=(const StringBag &otherSB) {
  Node *curr = otherSB.first;

  while (curr) {
    Node *node = this->find(curr->s);

    if (!node) {
      continue;
    }
    this->numElements -= min(node->arity, curr->arity);
    node->arity = max((size_t)0, node->arity - curr->arity);

    if (node->arity == 0) {
      this->numValues--;
    }

    curr = curr->next;
  }

  return *this;
}

bool StringBag::operator==(const StringBag &otherSB) const {
  if (this->numElements != otherSB.numElements ||
      this->numValues != otherSB.numValues)
    return false;

  Node *n1 = this->first;

  while (n1) {
    Node *n2 = otherSB.find(n1->s);

    if (!n2 || n2->arity != n1->arity) {
      return false;
    }
  }

  return true;
}

// go thru and delete Nodes with arity==0
void StringBag::dezombify() {
  Node *prev;
  Node *curr = this->first;

  while (curr) {
    if (curr->arity == 0) {
      if (prev) {
        prev->next = curr->next;
      } else {
        this->first = curr->next;
      }
      Node *tmp = curr->next;

      delete curr;
      curr = tmp;
    } else {
      prev = curr;
      curr = curr->next;
    }
  }
}

// Returns a pointer to the Node for value s if it exists, even if it has
// arity zero
StringBag::Node *StringBag::find(const string &s) const {
  Node *curr = this->first;

  while (curr && curr->s != s) {
    curr = curr->next;
  }

  return curr;
}

size_t StringBag::getNumElements() const { return this->numElements; }
size_t StringBag::getNumValues() const { return this->numValues; }
size_t StringBag::arity(const string &s) const {
  Node *node = this->find(s);
  if (!node) {
    return 0;
  }
  return node->arity;
}
