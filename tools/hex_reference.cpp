#include "etazero/game.h"
#include <iostream>

int main() {
    int size, count;
    while (std::cin >> size >> count) {
        etazero::Game game(size, 15, etazero::Rule::HEX);
        std::vector<int> moves, winners;
        for (int i = 0; i < count; ++i) {
            int action;
            std::cin >> action;
            game.play(action / size * 15 + action % size);
            moves.push_back(action);
            winners.push_back(game.winner());
        }
        const auto output = [&](const auto& values) {
            std::cout << '[';
            for (size_t i = 0; i < values.size(); ++i) std::cout << (i ? "," : "") << values[i];
            std::cout << ']';
        };
        auto obs = game.observation();
        std::vector<float> spatial(obs.begin(), obs.begin() + 1125);
        if (game.player() == -1) for (int plane = 0; plane < 5; ++plane)
            for (int row = 0; row < 15; ++row) for (int col = row + 1; col < 15; ++col)
                std::swap(spatial[plane * 225 + row * 15 + col], spatial[plane * 225 + col * 15 + row]);
        std::cout << "{\"size\":" << size << ",\"player\":" << game.player() << ",\"moves\":";
        output(moves);
        std::cout << ",\"winners\":"; output(winners);
        std::cout << ",\"board\":"; output(game.board().cells);
        std::cout << ",\"spatial\":"; output(spatial);
        std::cout << ",\"globals\":"; output(std::vector<float>(obs.begin() + 1125, obs.end()));
        std::cout << "}\n";
    }
}
