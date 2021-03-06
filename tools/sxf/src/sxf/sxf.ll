%{

#include <string.h>
#include "y.tab.h"
#include "sxf/SXF.hh"

void yyreset() { SXF::AcceleratorReader::s_reader->echo().set_lineno();} 

%}

%%

accelerator 	return SXF_ACCELERATOR;
sequence 	return SXF_SEQUENCE;
endsequence     return SXF_ENDSEQUENCE;
tag	   	return SXF_DESIGN;  
body	 	return SXF_BODY;
body.dev	return SXF_BODY_DEV;
at  	 	return SXF_AT;
arc		return SXF_ARC;	
l	 	return SXF_L;
n               return SXF_N; 


\/\/.*\n        ;

[a-zA-Z][a-zA-Z0-9_:.-]*   {
                  yylval.strval = (char *) malloc(strlen((char*) yytext) + 1);
                  strcpy(yylval.strval, (char*) yytext);
                  return SXF_IDENTIFIER;
                }

-?"."[0-9]+([eE][+-]?[0-9]+)?[lLfF]?  |
-?[0-9]+"."[0-9]*([eE][+-]?[0-9]+)?[lLfF]?  |
-?[0-9]+[eE][+-]?[0-9]+[lLfF]? |
-?[1-9][0-9]+ |
-?[0-9]		{
                  yylval.dval = atof((char*)yytext);
                  return SXF_NUMERIC_LITERAL;
                }

[ \t]*          ;

^.*\n		{
		  SXF::AcceleratorReader::s_reader->echo().increment_lineno(); 
		  SXF::AcceleratorReader::s_reader->echo().write_line((char *)yytext); 
		  REJECT;
                }

\n              ;

. 		{ return yytext[0];}

%%

