#include <iostream>
#include <vector>
#include <string>
#include <map>

using namespace std;

void eliminateLeftRecursion() {
    int numNonTerminals;
    cout << "Enter number of non-terminals: ";
    cin >> numNonTerminals;
    cin.ignore();

    vector<string> nonTerminals;
    map<string, vector<string>> productions;

    cout << "Enter non-terminals one by one:" << endl;
    for (int i = 0; i < numNonTerminals; ++i) {
        string nt;
        cout << "Non-terminal " << i + 1 << ": ";
        getline(cin, nt);
        nonTerminals.push_back(nt);
        productions[nt] = vector<string>();
    }

    cout << "\nEnter '^' for null (epsilon)" << endl;

    for (const string& nt : nonTerminals) {
        int pCount;
        cout << "Number of productions for " << nt << ": ";
        cin >> pCount;
        cin.ignore();

        cout << "Enter " << pCount << " productions for " << nt << ":" << endl;
        for (int i = 0; i < pCount; ++i) {
            string rhs;
            cout << "RHS of production " << i + 1 << ": ";
            getline(cin, rhs);
            productions[nt].push_back(rhs);
        }
    }

    map<string, vector<string>> newProductions;

    for (const string& nt : nonTerminals) {
        vector<string> alphas;
        vector<string> betas;

        bool isLeftRecursive = false;
        for (const string& rhs : productions[nt]) {
            if (rhs.find(nt) == 0) {
                alphas.push_back(rhs.substr(nt.size()));
                isLeftRecursive = true;
            } else {
                betas.push_back(rhs);
            }
        }

        if (isLeftRecursive) {
            string newNt = nt + "'";

            if (!betas.empty()) {
                for (const string& beta : betas) {
                    newProductions[nt].push_back(beta + newNt);
                }
            } else {
                cout << "Warning: Non-terminal " << nt << " has only left-recursive rules and no base case. It may be unreachable." << endl;
            }

            for (const string& alpha : alphas) {
                newProductions[newNt].push_back(alpha + newNt);
            }
            newProductions[newNt].push_back("^");
        } else {
            newProductions[nt] = productions[nt];
        }
    }

    cout << "\n----------------------------------------" << endl;
    cout << "Grammar after Eliminating Left Recursion" << endl;
    cout << "----------------------------------------" << endl;

    cout << "\nNew set of non-terminals: ";
    for (const auto& entry : newProductions) {
        cout << entry.first << " ";
    }
    cout << endl;

    cout << "\nNew set of productions:" << endl;
    for (const auto& entry : newProductions) {
        const string& nt = entry.first;
        const vector<string>& rhsList = entry.second;
        for (const string& rhs : rhsList) {
            cout << nt << " -> " << rhs << endl;
        }
    }
}

int main() {
    eliminateLeftRecursion();
    return 0;
}
