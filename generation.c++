// Include necessary headers
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>

// Define the Tokenizer class
class Tokenizer {
public:
    Tokenizer(const std::string& vocab_file, const std::string& merges_file) {
        load_vocab(vocab_file);
        load_merges(merges_file);
    }

    std::vector<std::string> tokenize(const std::string& text);
    std::vector<int> convert_tokens_to_ids(const std::vector<std::string>& tokens);
    std::string id_to_token(int token_id);

private:
    void load_vocab(const std::string& vocab_file);
    void load_merges(const std::string& merges_file);
    std::string bpe(const std::string& token);

    std::unordered_map<std::string, int> vocab;
    std::unordered_map<int, std::string> id_to_token_map;
    std::vector<std::pair<std::string, std::string>> merges;
    std::unordered_map<std::string, std::string> bpe_cache;
};

// Define the Llama class (LLM generation)
class Llama {
public:
    Llama(const std::string& model_path, const std::string& vocab_file, const std::string& merges_file) 
        : tokenizer_(vocab_file, merges_file) {
        // Load the model
        model_ = torch::jit::load(model_path);
    }

    std::vector<std::string> generate(
        const std::vector<std::string>& prompts,
        int max_gen_len,
        float temperature,
        float top_p) {
        // Tokenize and convert prompts to IDs
        std::vector<int64_t> tokenized_prompts;
        for (const auto& prompt : prompts) {
            auto tokens = tokenizer_.tokenize(prompt);
            auto ids = tokenizer_.convert_tokens_to_ids(tokens);
            tokenized_prompts.insert(tokenized_prompts.end(), ids.begin(), ids.end());
        }

        // Convert prompts to tensor
        torch::Tensor input_tensor = torch::tensor(tokenized_prompts, torch::dtype(torch::kInt64)).unsqueeze(0); // Batch size of 1

        // Prepare model input
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        inputs.push_back(torch::tensor(max_gen_len));
        inputs.push_back(torch::tensor(temperature));
        inputs.push_back(torch::tensor(top_p));

        // Run the model
        torch::Tensor outputs = model_.forward(inputs).toTensor();

        // Extract output tokens and convert to strings
        std::vector<std::string> output;
        for (int i = 0; i < outputs.size(1); i++) {
            int token_id = outputs[0][i].item<int>();
            output.push_back(tokenizer_.id_to_token(token_id));
        }

        return output;
    }

private:
    torch::jit::script::Module model_;
    Tokenizer tokenizer_;
};

// Tokenizer implementation
std::vector<std::string> Tokenizer::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : text) {
        if (c == ' ') {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        } else {
            token += c;
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<int> Tokenizer::convert_tokens_to_ids(const std::vector<std::string>& tokens) {
    std::vector<int> ids;
    for (const auto& token : tokens) {
        if (vocab.count(token)) {
            ids.push_back(vocab.at(token));
        } else {
            ids.push_back(vocab.at("<unk>")); // Unknown token
        }
    }
    return ids;
}

std::string Tokenizer::id_to_token(int token_id) {
    if (id_to_token_map.count(token_id)) {
        return id_to_token_map.at(token_id);
    } else {
        return "<unk>"; // Unknown token
    }
}

void Tokenizer::load_vocab(const std::string& vocab_file) {
    std::ifstream fin(vocab_file);
    std::string line;
    int id = 0;
    while (std::getline(fin, line)) {
        vocab[line] = id;
        id_to_token_map[id] = line;
        id++;
    }
}

void Tokenizer::load_merges(const std::string& merges_file) {
    std::ifstream fin(merges_file);
    std::string line;
    while (std::getline(fin, line)) {
        size_t space_pos = line.find(' ');
        std::string token1 = line.substr(0, space_pos);
        std::string token2 = line.substr(space_pos + 1);
        merges.push_back({token1, token2});
    }
}

// Define the EncryptDecrypt class
class EncryptDecrypt {
public:
    EncryptDecrypt(const std::string& file) : file_(file) {}

    void encrypt(int key);
    void decrypt(int key);

private:
    std::string file_;
};

void EncryptDecrypt::encrypt(int key) {
    std::ifstream fin(file_);
    std::ofstream fout("encrypt.txt");

    char c;
    while (fin >> std::noskipws >> c) {
        int temp = (c + key) % 256; // Ensure we stay within the bounds of char
        fout << static_cast<char>(temp);
    }

    fin.close();
    fout.close();
}

void EncryptDecrypt::decrypt(int key) {
    std::ifstream fin("encrypt.txt");
    std::ofstream fout("decrypt.txt");

    char c;
    while (fin >> std::noskipws >> c) {
        int temp = (c - key + 256) % 256; // Ensure we stay within the bounds of char
        fout << static_cast<char>(temp);
    }

    fin.close();
    fout.close();
}

int main() {
    Llama model("../model/model_scripted.pt", "../data/vocab.txt", "../data/merges.txt");
    std::vector<std::string> prompts = {"Hello, ", "World!"};
    int max_gen_len = 10;
    float temperature = 0.6;
    float top_p = 0.9;
    std::vector<std::string> output = model.generate(prompts, max_gen_len, temperature, top_p);

    for (const auto& word : output) {
        std::cout << word << " ";
    }
    std::cout << std::endl;

    EncryptDecrypt enc("S-op.txt");
    int key;
    std::cout << "Enter the key: ";
    std::cin >> key;
    enc.encrypt(key);
    enc.decrypt(key);

    return 0;
}
