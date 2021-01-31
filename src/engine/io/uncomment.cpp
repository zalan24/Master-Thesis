#include "uncomment.h"

bool uncomment(std::istream& in, std::ostream& out) {
    bool inComment = false;
    bool inString = false;
    bool inChar = false;
    bool lineComment = false;
    bool wasInComment = false;
    char lastC = '\0';
    char c;
    while (in.get(c)) {
        const bool canWrite = !wasInComment;
        wasInComment = inComment;
        switch (c) {
            case '\'':
                if (!inString && !inComment && (!inChar || lastC != '\\'))
                    inChar = !inChar;
                break;
            case '"':
                if (!inChar && !inComment && (!inString || lastC != '\\'))
                    inString = !inString;
                break;
            case '/':
                if (lastC == '/' && !inChar && !inString && !inComment) {
                    inComment = true;
                    lineComment = true;
                }
                else if (lastC == '*' && !inChar && !inString && inComment && !lineComment) {
                    inComment = false;
                }
                break;
            case '*':
                if (lastC == '/' && !inChar && !inString && !inComment) {
                    inComment = true;
                    lineComment = false;
                }
                break;
            case '\n':
                if (inComment && lineComment)
                    inComment = false;
                break;
            default:
                break;
        }
        if (lastC != '\0' && !inComment && canWrite) {
            out.put(lastC);
        }
        lastC = c;
    }
    if (lastC != '\0' && !inComment && !wasInComment)
        out.put(lastC);
    return !inString && !inChar;
}
