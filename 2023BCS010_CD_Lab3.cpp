#include <bits/stdc++.h>
using namespace std;

int main() {
    int nStates, nSymbols;

    cout << "Enter number of NFA states: ";
    cin >> nStates;

    cout << "Enter number of input symbols: ";
    cin >> nSymbols;

    vector<string> symbols(nSymbols);
    cout << "Enter the input symbols separated by space: ";
    for(int i=0;i<nSymbols;i++) cin >> symbols[i];

    // NFA transitions
    map<string,map<string,vector<string>>> nfa;
    vector<string> nfaStates;

    cout << "\n--- Enter transitions for each state ---\n";
    for(int i=0;i<nStates;i++){
        string state;
        cout << "\nState name: ";
        cin >> state;
        nfaStates.push_back(state);

        for(auto &sym: symbols){
            cout << "  End states from state " << state
                 << " on input '" << sym
                 << "' (space-separated or '-' for none): ";
            string line;
            getline(cin >> ws, line);
            vector<string> ends;
            if(line!="-"){
                stringstream ss(line);
                string s;
                while(ss >> s) ends.push_back(s);
            }
            nfa[state][sym] = ends;
        }
    }

    cout << "\nEnter final NFA states (space-separated): ";
    string finalsLine;
    getline(cin >> ws, finalsLine);
    set<string> nfaFinal;
    {
        stringstream ss(finalsLine);
        string s;
        while(ss >> s) nfaFinal.insert(s);
    }

    // ---------- Subset construction ----------
    set<set<string>> dfaStates;
    queue<set<string>> q;
    set<string> start; start.insert(nfaStates[0]); // take first NFA state as start
    dfaStates.insert(start); q.push(start);

    map<set<string>, map<string,set<string>>> dfa;
    set<set<string>> dfaFinal;

    while(!q.empty()){
        auto cur = q.front(); q.pop();
        for(auto &sym: symbols){
            set<string> next;
            for(auto &st: cur){
                for(auto &to: nfa[st][sym]) next.insert(to);
            }
            dfa[cur][sym] = next;
            if(!next.empty() && !dfaStates.count(next)){
                dfaStates.insert(next);
                q.push(next);
            }
        }
    }

    // Determine DFA final states
    for(auto &stset: dfaStates){
        for(auto &f: nfaFinal){
            if(stset.count(f)) { dfaFinal.insert(stset); break; }
        }
    }

    // ---------- Output ----------
    cout << "\n========== DFA Transition Table ==========\n";
    for(auto &stset: dfaStates){
        cout << "{";
        bool first=true;
        for(auto &s: stset){ if(!first) cout<<","; cout<<s; first=false; }
        cout << "}";
        for(auto &sym: symbols){
            cout << " --" << sym << "--> {";
            bool f=true;
            for(auto &s: dfa[stset][sym]){ if(!f) cout<<","; cout<<s; f=false; }
            cout << "}";
        }
        cout << "\n";
    }

    // ---------- Print final DFA states on a separate line ----------
    cout << "\nFinal DFA states: ";
    for(auto &stset: dfaFinal){
        cout << "{";
        bool first=true;
        for(auto &s: stset){ if(!first) cout<<","; cout<<s; first=false; }
        cout << "} ";
    }
    cout << "\n";
    return 0;
}
