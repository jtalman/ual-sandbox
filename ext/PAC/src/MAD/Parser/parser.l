%{

#include <string.h>

int len;
int lineno;
char linebuf[256];

yyreset() { lineno = 1; } 

#define RETURN_TOKEN(token) { return(token); }

#define RETURN_STR(token)                            \
     { int len = strlen((char *)yytext)+1;           \
        yylval.Str = (char *)malloc((unsigned)len);  \
        strcpy(yylval.Str, (char *)yytext);          \
        return(token); }

#define RETURN_REAL(token)  \
     { yylval.Real = atof((char *)yytext); return(token); }

#define RETURN_CHR(token)  \
     { yylval.Chr  = ((char *)yytext)[0];  return(token); }
%}

D [0-9]
E [DEde][-+]?{D}+
L [a-z]

%%
\,                     {}
[ ]+                   {}
\(                     { RETURN_STR(O_PARENTHESES)}
\)                     { RETURN_STR(C_PARENTHESES)}
\[                     { RETURN_TOKEN(O_BRACKET)}
\]                     { RETURN_TOKEN(C_BRACKET)}
[:]                    { RETURN_TOKEN(COLON)}
^[ \t]+                {}
\n                     { RETURN_TOKEN(LINE_END) }

;                      { RETURN_CHR(STATEMENT_END) }
[!].*                  { RETURN_STR(COMMENTS) }
"return".*             { RETURN_STR(COMMENTS) }
"title".*              { RETURN_STR(COMMENTS) }

[:][ ]*"line"[ ]*[=]   { RETURN_TOKEN(LINE) }

[:][ ]*"sequ"{L}*      { RETURN_TOKEN(SEQUENCE) }
"endsequence"          { RETURN_TOKEN(ENDSEQUENCE) }
"refer"[ ]*[=]         { RETURN_TOKEN(REFER) }
"at"[ ]*[=]            { RETURN_TOKEN(AT) }

"tilt" |
"t"{D}                 { RETURN_STR(TILT) }
[a-z][a-z0-9._]*       { RETURN_STR(LABEL) }

[:]{L}*[=]             { RETURN_TOKEN(PARAMETER); }

{D}+ |
{D}+"."{D}*({E})? |
{D}*"."{D}+({E})? |
{D}+{E}                { RETURN_STR(NUM) }

[\+\-\/\*]             { RETURN_STR(EXP) }

"sqrt"[ ]*\(  |
"cos"[ ]*\(  |
"sin"[ ]*\(  |
"asin"[ ]*\( |
"tan"[ ]*\(  |
"abs"[ ]*\(            { RETURN_STR(FNCT) }

"type"[ ]*[=]          { RETURN_CHR(TYPE) }
[a-z][a-z0-9_]*[ ]*[=] { RETURN_STR(ATTRIBUTE) }

<<EOF>>                { printf("1;\n"); exit(0); }

%%

