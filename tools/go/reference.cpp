#include "bridge.cpp"
#include "external/nlohmann_json/json.hpp"
#include <iostream>
int main() {
    using nlohmann::json;
    json cases = json::array();
    for (int size : {9,13,19}) {
        GoState* game = go_create(size,7.5f);
        std::vector<int> actions;
        auto record = [&]() {
            const int* p = go_snapshot(game);
            int area = size*size;
            const float* obs = go_observation(game);
            json spatial = json::array();
            for (int i = 0; i < 22*361; i++) if (obs[i] != 0) spatial.push_back({i,obs[i]});
            cases.push_back({{"size",size},{"actions",actions},{"player",p[0]},{"finished",bool(p[1])},{"winner",p[2]},
                {"board",std::vector<int>(p+4,p+4+area)},{"legal",std::vector<int>(p+5+area,p+5+area+p[4+area])},
                {"score",go_score(game)},{"spatial",spatial},{"globals",std::vector<float>(obs+22*361,obs+22*361+19)}});
        };
        record();
        for (int a : {1,0,size,size*size,size+1}) {
            if (!go_play(game,a)) return 1;
            actions.push_back(a);
            record();
        }
        for (int i=0;i<65;i++) {
            const int* p = go_snapshot(game);
            int n = p[4+size*size];
            int a = p[5+size*size+(i*37+11)%(n-1)];
            if (!go_play(game,a)) return 2;
            actions.push_back(a);
        }
        record();
        for (int i=0;i<2;i++) { go_play(game,size*size); actions.push_back(size*size); }
        record();
        go_delete(game);
    }
    std::cout << json({{"cases",cases}}).dump() << std::endl;
}
