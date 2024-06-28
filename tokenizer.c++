#include "tokenizer.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

Tokenizer::Tokenizer(const std::string& vocab_file, const std::string& merges_file) {
    load_vocab(vocab_file);
    load_merges(merges_file);
}

void Tokenizer::load_vocab(const std::string& vocab_file) {
    std::ifstream file(vocab_file);
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        vocab[line] = index++;
    }
}

void Tokenizer::load_merges(const std::string& merges_file) {
    std::ifstream file(merges_file);
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token1, token2;
        iss >> token1 >> token2;
        merges.push_back({token1, token2});
    }
}

std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::istringstream stream(text);
    std::string word;
    while (stream >> word) {
        tokens.push_back(bpe(word));
    }
    return tokens;
}

std::string Tokenizer::bpe(const std::string& token) {
    if (bpe_cache.find(token) != bpe_cache.end()) {
        return bpe_cache[token];
    }

    std::vector<std::string> chars(token.begin(), token.end());
    while (chars.size() > 1) {
        bool found = false;
        for (const auto& merge : merges) {
            std::string pair = merge.first + merge.second;
            auto it = std::search(chars.begin(), chars.end(), pair.begin(), pair.end());
            if (it != chars.end()) {
                chars.erase(it, it + pair.size());
                chars.insert(it, pair);
                found = true;
                break;
            }
        }
        if (!found) {
            break;
        }
    }

    std::string result = "";
    for (const auto& c : chars) {
        result += c;
    }

    bpe_cache[token] = result;
    return result;
}

std::vector<int> Tokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens) {
    std::vector<int> ids;
    for (const auto& token : tokens) {
        if (vocab.find(token) != vocab.end()) {
            ids.push_back(vocab[token]);
        } else {
            ids.push_back(vocab["<unk>"]);
        }
    }
    return ids;
}
