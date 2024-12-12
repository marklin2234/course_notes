module tierlist;
import <iostream>;
import <utility>;
import <algorithm>;

void TierList::swap(TierList &other) {
  std::swap(tiers, other.tiers);
  std::swap(tierCount, other.tierCount);
  std::swap(reserved, other.reserved);
}

void TierList::enlarge() {
  reserved = 2 + 2 * std::max(reserved, tierCount);
  List **newTiers = new List *[reserved];
  std::fill_n(newTiers, reserved, nullptr);
  std::copy_n(tiers, tierCount, newTiers);
  delete[] tiers;
  tiers = newTiers;
}

TierList::TierList() : tiers{nullptr}, tierCount{0}, reserved(0) {}
TierList::~TierList() {
  if (tiers) {
    for (size_t i = 0; i < tierCount; ++i) {
      delete tiers[i];
    }
  }
  delete[] tiers;
}

void TierList::push_back_tier() {
  // Need at least 1 extra as sentinel.
  if (tierCount + 1 >= reserved) {
    enlarge();
  }
  tiers[tierCount++] = new List;
}
void TierList::pop_back_tier() {
  if (tierCount > 0) {
    delete tiers[--tierCount];
    tiers[tierCount] = nullptr;
  }
}

void TierList::push_front_at_tier(size_t tier, const std::string &entry) {
  tiers[tier]->push_front(entry);
}
void TierList::pop_front_at_tier(size_t tier) {
  tiers[tier]->pop_front();
}

size_t TierList::tierSize() const { return tierCount; }
size_t TierList::size() const {
  size_t result = 0;
  for (size_t i = 0; i < tierCount; i++) {
    result += tiers[i]->size();
  }
  return result;
}

TierList::Iterator::Iterator(const TierList *tierList_) : tierList{tierList_}, tier{0}, isEnd{false} {
    while(tier < tierList->tierCount && tierList->tiers[tier]->size() == 0) {
        tier++;
    }

    if (tier == tierList->tierCount) {
        isEnd = true;
    }
    std::cout << tier << "\n";
}

TierList::Iterator::Iterator(const TierList *tierList_, bool isEnd_) : tierList{tierList_}, tier{0}, isEnd{isEnd_} {}

TierList::value_type TierList::Iterator::operator*() const {
    TierList::value_type ret = { tier, *tierList->tiers[tier]->begin() };

    return ret;
}

TierList::Iterator &TierList::Iterator::operator++() {
    tier++;
    while(tier < tierList->tierCount && tierList->tiers[tier]->size() == 0) {
        tier++;
    }
    if (tier == tierList->tierCount) {
        isEnd = true;
    }

    return *this;
}

TierList::Iterator TierList::Iterator::operator<<(int bk) const {
    if (tier - bk < 0) {
        TierList::Iterator newIterator(*this);
        newIterator.isEnd = true;
        return newIterator;
    }
    TierList::Iterator newIterator(*this);
    newIterator.tier -= bk;
    while(newIterator.tier > 0 && tierList->tiers[newIterator.tier]->size() == 0) {
        newIterator.tier--;
    }

    if (newIterator.tier == 0 && tierList->tiers[newIterator.tier]->size() == 0) {
        newIterator.isEnd = true;
    }

    return newIterator;
}

TierList::Iterator TierList::Iterator::operator>>(int fwd) const {
    if (tier + fwd >= tierList->tierCount) {
        TierList::Iterator newIterator(*this);
        newIterator.isEnd = true;
        return newIterator;
    }
    
    TierList::Iterator newIterator(*this);
    newIterator.tier += fwd;
    while(newIterator.tier < tierList->tierCount && tierList->tiers[newIterator.tier]->size() == 0) {
        newIterator.tier++;
    }

    if (newIterator.tier == tierList->tierCount) {
        newIterator.isEnd = true;
    }

    std::cout << newIterator.tier << "\n";
    return newIterator;
}

bool TierList::Iterator::operator!=(const Iterator &other) const {
    return this->tierList == other.tierList && this->tier == other.tier && this->isEnd == other.isEnd;
}

TierList::Iterator TierList::begin() const {
    return TierList::Iterator(this);
}

TierList::Iterator TierList::end() const {
    return TierList::Iterator(this, true);
}

