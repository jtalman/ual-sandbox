%{
#include  <string.h>
#include <math.h>
#include <malloc.h>

int line_node_position = 0;

#define PRINT_COMMENT(comment)         \
  { printf("%s", comment);             \
    free(comment); }

#define SAVE_STATEMENT_END(where) \
 { where =  (char *) malloc((unsigned) 1); \
   sprintf(where, "");} 

#define SAVE_LINE_END(where)      \
 { where =  (char *) malloc((unsigned) 2); \
   sprintf(where, "\n");} 

#define SAVE_COMMENTS(where, comments)                      \
 { where = (char *) malloc((unsigned) (strlen(comments) + 5));       \
   sprintf(where, "# %s \n", comments); free(comments); } 
 
#define PRINT_PARAMETER(label, exp, end )       \
  { printf("my $%s = %s; %s", label, exp, end); \
  free(label); free(exp); free(end);} 

#define SAVE_NUM(where, number)                   \
 { where = (char *) malloc((unsigned) (strlen(number) + 2)); \
   sprintf(where, "%s", number); free(number); } 

#define SAVE_LABEL(where, label)                   \
 { where = (char *) malloc((unsigned) (strlen(label) + 2)); \
   sprintf(where, "$%s", label); free(label); } 

#define SAVE_BRACKETS(where, element, attrib)                             \
  { where = (char *) malloc((unsigned) (strlen(element) + strlen(attrib) + 30));   \
    sprintf(where, "$shell->attribute(\"%s\", \"%s\")", element, attrib);   \
    free(element); free(attrib); } 

#define SAVE_EXP(where, left, exp, right)                                        \
  { where = (char *) malloc((unsigned) (strlen(left) + strlen(exp) + strlen(right) + 3)); \
    sprintf(where, "%s %s %s", left, exp, right);                                \
    free(left); free(exp); free(right);} 

#define SAVE_FNCT(where, fnct, exp)                                        \
  { where = (char *) malloc((unsigned) (strlen(fnct) + strlen(exp) + 12)); \
    sprintf(where, "POSIX::%s %s )", fnct, exp);                           \
    free(fnct); free(exp); } 

#define SAVE_UMINUS(where, exp, right)                             \
  { where = (char *) malloc((unsigned) ( strlen(exp) + strlen(right) + 2)); \
    sprintf(where, "%s %s", exp, right);                           \
    free(exp); free(right);} 


#define PRINT_ELEMENT(label, keyword, attributes, end)                               \
  { printf("$shell->element(\"%s\", %s, {%s}); %s ", label, keyword, attributes, end); \
    free(label); free(keyword); free(attributes); free(end); } 

#define SAVE_KEYWORD(where, keyword)                  \
  { where = (char *) malloc((unsigned) (strlen(keyword) + 3)); \
    sprintf(where, "\"%s\"", keyword);                \
    free(keyword); }

#define SAVE_EMPTINESS(where)                       \
   { where = (char *) malloc((unsigned) 1);         \
     where[0] = '\0'; }

#define SAVE_TILT(where, attributes, t)                                          \
   { where = (char *) malloc((unsigned) (strlen(attributes) + strlen(t) + 15));  \
     sprintf(where, "%s tilt => \"%s\",", attributes, t);                        \
     free(t); free(attributes); }

#define SAVE_TYPE(where, attributes, t)                                          \
   { where = (char *) malloc((unsigned) (strlen(attributes) + strlen(t) + 15));  \
     sprintf(where, "%s type => \"%s\",", attributes, t);                        \
     free(t); free(attributes); }

#define SAVE_ATTRIBUTES(where, attributes, attribute, exp)                                           \
   { where = (char *) malloc((unsigned) (strlen(attributes) + strlen(attribute) + strlen(exp) + 9)); \
     attribute[strlen(attribute)-1] = ' ';                                                           \
     sprintf(where, "%s %s => %s,\0", attributes, attribute, exp);                                   \
     free(attributes), free(attribute); free(exp); }

#define PRINT_LINE(label, nodes, end)                             \
   { printf("$shell->line(\"%s\", \n%s); %s", label, nodes, end); \
     free(label); free(nodes); free(end); } 

#define SAVE_LINE_NODES(where, nodes, node)                                   \
   { where = (char *) malloc((unsigned) (strlen(nodes) + strlen(node) + 7));  \
     if(strlen(nodes) + strlen(node) < 70) { line_node_position = 0; }        \
     if(strlen(nodes) + strlen(node) > line_node_position + 70) \
       { sprintf(where, "%s \"%s\",\n\0", nodes, node);  line_node_position = strlen(nodes) + strlen(node) + 4;  } \
     else  \
       { sprintf(where, "%s \"%s\",\0", nodes, node); }   \
     free(nodes), free(node);}

#define PRINT_O_SEQUENCE(label, exp, end)                                     \
   { printf("$sequence = $shell->sequence(\"%s\",{refer => \"%s\"}); $sequence->set( %s ", label, exp, end); \
     free(label); free(exp); free(end); } 

#define PRINT_C_SEQUENCE(end) \
   { printf("); %s ", end);   \
     free(end); } 

#define PRINT_NAMED_SEQ_NODE(name, keyword, exp, attributes, end)                        \
  { printf("[\"%s\", %s, {at => %s}, { %s}], %s ", name, keyword, exp, attributes, end); \
    free(name); free(keyword); free(exp), free(attributes); free(end); } 

#define PRINT_SIMPLE_SEQ_NODE(keyword, exp, attributes, end)                         \
  { printf("[\"\", \"%s\", {at => %s}, { %s}], %s ", keyword, exp, attributes, end); \
    free(keyword); free(exp), free(attributes); free(end); } 


%}

%union {
     int    Int;
     char   Chr;
     char*  Str;
     double Real;
}

%token <Chr>  STATEMENT_END
%token <Chr>  LINE_END
%token <Chr>  COLON
%token <Chr>  O_BRACKET
%token <Chr>  C_BRACKET
%token <Str>  O_PARENTHESES
%token <Str>  C_PARENTHESES
%token <Str>  NUM
%token <Str>  LABEL
%token <Str>  EXP
%token <Str>  FNCT

%token <Str>  COMMENTS
%token <Chr>  PARAMETER
%token <Str>  ATTRIBUTE
%token <Str>  TILT
%token <Chr>  TYPE
%token <Chr>  LINE
%token <Chr>  SEQUENCE
%token <Chr>  ENDSEQUENCE
%token <Chr>  AT
%token <Chr>  REFER

%type  <Str>  statement_end
%type  <Str>  exp
%type  <Str>  keyword
%type  <Str>  attributes
%type  <Str>  line_nodes

%left EXP
%left UMINUS

%% /* Grammar rules and actions follow */

input : 
      | input definition
;

definition  : comment {}
            | parameter {}
            | element {}
            | line {}
            | o_sequence {}
            | seq_node {}
            | c_sequence {}
;


statement_end       : STATEMENT_END                { SAVE_STATEMENT_END($$) }
                    | LINE_END                     { SAVE_LINE_END($$) }
                    | COMMENTS LINE_END            { SAVE_COMMENTS($$, $1) } 
;

/* Comments */

comment             : statement_end                { PRINT_COMMENT($1) }
;

/* Parameters */

parameter : LABEL PARAMETER exp statement_end { PRINT_PARAMETER($1, $3, $4) } 
;

exp       : NUM                                   { SAVE_NUM($$, $1) }
          | LABEL                                 { SAVE_LABEL($$, $1) } 
          | LABEL O_BRACKET LABEL C_BRACKET       { SAVE_BRACKETS($$, $1, $3) }  
          | exp EXP exp                           { SAVE_EXP($$, $1, $2, $3) } 
          | O_PARENTHESES exp C_PARENTHESES       { SAVE_EXP($$, $1, $2, $3) } 
          | EXP exp %prec UMINUS                  { SAVE_UMINUS($$, $1, $2) }  
          | FNCT exp  C_PARENTHESES               { SAVE_FNCT($$, $1, $2) }    
;

/* Elements */

element    : LABEL keyword attributes statement_end { PRINT_ELEMENT($1, $2, $3, $4) }
;

keyword    : COLON LABEL                            { SAVE_KEYWORD($$, $2) }
;

attributes :                                        { SAVE_EMPTINESS($$) }
           | attributes ATTRIBUTE exp               { SAVE_ATTRIBUTES($$, $1, $2, $3) }
           | attributes TILT                        { SAVE_TILT($$, $1, $2) }   
           | attributes TYPE LABEL                  { SAVE_TYPE($$, $1, $3) } 
;

/* Line */

line : LABEL LINE O_PARENTHESES  line_nodes C_PARENTHESES statement_end { PRINT_LINE($1, $4, $6) }
;

line_nodes    :                                    { SAVE_EMPTINESS($$) }
              | line_nodes LABEL                   { SAVE_LINE_NODES($$, $1, $2) }
;

/* Sequence */

o_sequence    : LABEL SEQUENCE REFER LABEL statement_end      { PRINT_O_SEQUENCE($1, $4, $5) }
;

c_sequence    : ENDSEQUENCE statement_end                     { PRINT_C_SEQUENCE($2) }
;

seq_node      :       LABEL  AT exp attributes statement_end  { PRINT_SIMPLE_SEQ_NODE($1, $3, $4, $5) }     
              | LABEL keyword AT exp attributes statement_end { PRINT_NAMED_SEQ_NODE($1, $2, $4, $5, $6) }
;



%%

main () { yyparse (); }
yyerror (char* s) { printf ("%s\n", s); }

#include "lex.yy.c"

