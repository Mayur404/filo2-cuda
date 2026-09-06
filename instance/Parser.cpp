#include "Parser.hpp"

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

// fscanf performs a formatted-library call for every scalar in a million
// customer instance. Keep the same token-oriented grammar, but amortize I/O
// through a large buffer and parse the small set of numeric fields locally.
class FastScanner {
public:
    explicit FastScanner(FILE* file) : file_(file) { }

    bool next_token(char* output, std::size_t capacity) {
        if (capacity == 0) return false;

        int character = get();
        while (character != EOF && is_space(character)) character = get();
        if (character == EOF) return false;

        // Treat ':' as punctuation even when a TSPLIB producer writes
        // "NAME:" instead of "NAME :".
        if (character == ':') {
            if (capacity < 2) return false;
            output[0] = ':';
            output[1] = '\0';
            return true;
        }

        std::size_t length = 0;
        bool fits = true;
        do {
            if (character == ':') {
                pending_ = character;
                break;
            }
            if (length + 1 < capacity) {
                output[length] = static_cast<char>(character);
            } else {
                fits = false;
            }
            ++length;
            character = get();
        } while (character != EOF && !is_space(character));

        // Keep the delimiter for skip_line(). This matters when a header
        // value is immediately followed by its newline: consuming that
        // newline here would make the subsequent line skip the next header.
        if (character != EOF) pending_ = character;

        if (fits) output[length] = '\0';
        return fits;
    }

    bool next_int(int* value) {
        char token[32];
        if (!next_token(token, sizeof(token))) return false;
        errno = 0;
        char* end = nullptr;
        const long parsed = std::strtol(token, &end, 10);
        if (errno == ERANGE || end == token || *end != '\0' ||
            parsed < INT_MIN || parsed > INT_MAX) {
            return false;
        }
        *value = static_cast<int>(parsed);
        return true;
    }

    bool next_double(double* value) {
        // TSPLIB coordinate values are short decimal tokens. 128 bytes also
        // leaves room for a signed exponent and spellings such as "nan".
        char token[128];
        if (!next_token(token, sizeof(token))) return false;
        errno = 0;
        char* end = nullptr;
        const double parsed = std::strtod(token, &end);
        if (errno == ERANGE || end == token || *end != '\0') return false;
        *value = parsed;
        return true;
    }

    void skip_line() {
        int character = get();
        while (character != EOF && character != '\n') character = get();
    }

private:
    static bool is_space(int character) {
        return character == ' ' || character == '\t' || character == '\n' ||
               character == '\r' || character == '\f' || character == '\v';
    }

    int get() {
        if (pending_ != EOF) {
            const int character = pending_;
            pending_ = EOF;
            return character;
        }
        if (position_ == size_) {
            size_ = std::fread(buffer_, 1, sizeof(buffer_), file_);
            position_ = 0;
            if (size_ == 0) return EOF;
        }
        return static_cast<unsigned char>(buffer_[position_++]);
    }

    FILE* file_;
    char buffer_[1 << 16];
    std::size_t position_ = 0;
    std::size_t size_ = 0;
    int pending_ = EOF;
};

class FileGuard {
public:
    explicit FileGuard(FILE* file) : file_(file) { }
    ~FileGuard() {
        if (file_ != nullptr) std::fclose(file_);
    }

private:
    FILE* file_;
};

}  // namespace

namespace cobra {

    Parser::Parser(const std::string& filepath) : filepath(filepath) { }

    std::optional<InstanceData> Parser::Parse() {

        FILE* file = fopen(filepath.c_str(), "r");

        if (!file) {
            return std::nullopt;
        }

        FileGuard file_guard(file);
        FastScanner scanner(file);
        InstanceData data;
        char token[128];

        const auto expect = [&scanner, &token](const char* expected) {
            return scanner.next_token(token, sizeof(token)) && std::strcmp(token, expected) == 0;
        };
        const auto expect_colon = [&scanner, &token]() {
            return scanner.next_token(token, sizeof(token)) && std::strcmp(token, ":") == 0;
        };

        if (!expect("NAME") || !expect_colon() || !scanner.next_token(token, sizeof(token))) {
            return std::nullopt;
        }
        scanner.skip_line();
        if (!expect("COMMENT") || !expect_colon()) return std::nullopt;
        scanner.skip_line();
        if (!expect("TYPE") || !expect_colon()) return std::nullopt;
        scanner.skip_line();

        int matrix_size = 0;
        if (!expect("DIMENSION") || !expect_colon() || !scanner.next_int(&matrix_size) || matrix_size <= 0) {
            return std::nullopt;
        }
        scanner.skip_line();
        if (!expect("EDGE_WEIGHT_TYPE") || !expect_colon() || !scanner.next_token(token, sizeof(token))) {
            return std::nullopt;
        }
        scanner.skip_line();

        if (!expect("CAPACITY") || !expect_colon() || !scanner.next_int(&data.vehicle_capacity)) {
            return std::nullopt;
        }
        scanner.skip_line();

        data.xcoords.resize(static_cast<std::size_t>(matrix_size));
        data.ycoords.resize(static_cast<std::size_t>(matrix_size));

        if (!expect("NODE_COORD_SECTION")) return std::nullopt;
        scanner.skip_line();

        int vertex_index = 0;
        for (int i = 0; i < matrix_size; ++i) {
            if (!scanner.next_int(&vertex_index) || !scanner.next_double(&data.xcoords[static_cast<std::size_t>(i)]) ||
                !scanner.next_double(&data.ycoords[static_cast<std::size_t>(i)])) {
                return std::nullopt;
            }
        }

        if (!expect("DEMAND_SECTION")) return std::nullopt;
        scanner.skip_line();

        data.demands.resize(static_cast<std::size_t>(matrix_size));
        for (int i = 0; i < matrix_size; ++i) {
            if (!scanner.next_int(&vertex_index) || !scanner.next_int(&data.demands[static_cast<std::size_t>(i)])) {
                return std::nullopt;
            }
        }

        return data;
    }

}  // namespace cobra