#include <iostream>
#include <memory>

// Base character class
class Character {
public:
    virtual ~Character() = default;
    virtual void action() const = 0;
};

// Concrete Character class representing Kirby
class Kirby : public Character {
public:
    void action() const override {
        std::cout << "Kirby performs a basic action." << std::endl;
    }
};

// Base decorator class
class Decorator : public Character {
protected:
    std::shared_ptr<Character> character;

public:
    Decorator(std::shared_ptr<Character> c) : character(c) {}

    virtual void action() const override {
        character->action();
    }
};

// Concrete decorator for Fire ability
class FireKirby : public Decorator {
public:
    FireKirby(std::shared_ptr<Character> c) : Decorator(c) {}

    void action() const override {
        Decorator::action();
        std::cout << "Kirby uses Fire ability!" << std::endl;
    }
};

// Concrete decorator for Ice ability
class IceKirby : public Decorator {
public:
    IceKirby(std::shared_ptr<Character> c) : Decorator(c) {}

    void action() const override {
        Decorator::action();
        std::cout << "Kirby uses Ice ability!" << std::endl;
    }
};

int main() {
    // Create a base Kirby character
    std::shared_ptr<Character> kirby = std::make_shared<Kirby>();

    // Kirby with Fire ability
    std::shared_ptr<Character> kirbyWithFire = std::make_shared<FireKirby>(kirby);
    kirbyWithFire->action();

    // Kirby with Ice ability
    std::shared_ptr<Character> kirbyWithIce = std::make_shared<IceKirby>(kirby);
    kirbyWithIce->action();

    // Kirby with both Fire and Ice abilities
    std::shared_ptr<Character> kirbyWithFireAndIce = std::make_shared<IceKirby>(
        std::make_shared<FireKirby>(kirby)
    );
    kirbyWithFireAndIce->action();

    return 0;
}