#include <bits/stdc++.h>
using namespace std;

struct Transition
{
    int fromState;
    char symbol;
    int toState;
};

struct NFA
{
    int startState;
    int finalState;
};

vector<Transition> v;
int totalStates = 0;

int precedence(char op)
{
    if (op == '*')
        return 3;
    if (op == '.')
        return 2;
    if (op == '|')
        return 1;
    return 0;
}

string addConcat(const string &s)
{
    string t = "";
    for (int i = 0; i < (int)s.length(); i++)
    {
        t += s[i];
        if (i + 1 < (int)s.length())
        {
            char curr = s[i];
            char next = s[i + 1];
            if ((isalnum((unsigned char)curr) || curr == ')' || curr == '*') &&
                (isalnum((unsigned char)next) || next == '('))
            {
                t += '.';
            }
        }
    }
    return t;
}

string infixToPostfix(const string &infix)
{
    string postfix = "";
    stack<char> st;
    string ss = addConcat(infix);

    for (char token : ss)
    {
        if (isalnum((unsigned char)token))
        {
            postfix += token;
        }
        else if (token == '(')
        {
            st.push(token);
        }
        else if (token == ')')
        {
            while (!st.empty() && st.top() != '(')
            {
                postfix += st.top();
                st.pop();
            }
            if (!st.empty())
                st.pop();
        }
        else
        {
            while (!st.empty() && precedence(st.top()) >= precedence(token))
            {
                postfix += st.top();
                st.pop();
            }
            st.push(token);
        }
    }

    while (!st.empty())
    {
        if (st.top() != '(')
            postfix += st.top();
        st.pop();
    }
    return postfix;
}

NFA createNFA(char symbol)
{
    NFA nfa;
    nfa.startState = totalStates++;
    nfa.finalState = totalStates++;
    v.push_back({nfa.startState, symbol, nfa.finalState});
    return nfa;
}

NFA unionNFA(const NFA &n1, const NFA &n2)
{
    NFA newNFA;
    newNFA.startState = totalStates++;
    newNFA.finalState = totalStates++;

    v.push_back({newNFA.startState, 'e', n1.startState});
    v.push_back({newNFA.startState, 'e', n2.startState});

    v.push_back({n1.finalState, 'e', newNFA.finalState});
    v.push_back({n2.finalState, 'e', newNFA.finalState});

    return newNFA;
}

NFA concatenateNFA(const NFA &n1, const NFA &n2)
{
    v.push_back({n1.finalState, 'e', n2.startState});

    NFA newNFA;
    newNFA.startState = n1.startState;
    newNFA.finalState = n2.finalState;
    return newNFA;
}

NFA kleeneStarNFA(const NFA &n)
{
    NFA newNFA;
    newNFA.startState = totalStates++;
    newNFA.finalState = totalStates++;

    v.push_back({newNFA.startState, 'e', n.startState});
    v.push_back({newNFA.startState, 'e', newNFA.finalState});

    v.push_back({n.finalState, 'e', n.startState});
    v.push_back({n.finalState, 'e', newNFA.finalState});

    return newNFA;
}

NFA postfixToNFA(const string &s)
{
    stack<NFA> st;

    for (char token : s)
    {
        if (isalnum((unsigned char)token))
        {
            st.push(createNFA(token));
        }
        else if (token == '|')
        {
            if (st.size() < 2)
                continue;
            NFA nfa2 = st.top();
            st.pop();
            NFA nfa1 = st.top();
            st.pop();
            st.push(unionNFA(nfa1, nfa2));
        }
        else if (token == '.')
        {
            if (st.size() < 2)
                continue;
            NFA nfa2 = st.top();
            st.pop();
            NFA nfa1 = st.top();
            st.pop();
            st.push(concatenateNFA(nfa1, nfa2));
        }
        else if (token == '*')
        {
            if (st.empty())
                continue;
            NFA nfa = st.top();
            st.pop();
            st.push(kleeneStarNFA(nfa));
        }
    }
    return st.top();
}

void displayNFA(const NFA &nfa)
{
    cout << "\nTransition function:\n";

    map<pair<int, char>, vector<int>> mp;
    for (const auto &transition : v)
    {
        mp[{transition.fromState, transition.symbol}].push_back(transition.toState);
    }

    for (const auto &pair : mp)
    {
        cout << "q[" << pair.first.first << "," << pair.first.second << "] --> ";
        for (int i = 0; i < (int)pair.second.size(); ++i)
        {
            cout << pair.second[i] << (i == (int)pair.second.size() - 1 ? "" : " & ");
        }
        cout << endl;
    }
    cout << "\nStart state: " << nfa.startState << endl;
    cout << "Final state: " << nfa.finalState << endl;
}

int main()
{
    string s;
    cout << "Enter the regular expression (use '|' for union): ";
    cin >> s;
    string postfix = infixToPostfix(s);
    NFA finalNFA = postfixToNFA(postfix);
    displayNFA(finalNFA);
    return 0;
}
