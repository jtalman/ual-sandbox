/* sxf.yy - YACC grammar for SXF */

%{

/* Declarations */

#include "sxf/SXF.hh"
#include <stdio.h>
#include <iostream>
#include <malloc.h>

void yyerror(char *);
int yylex (void);
extern "C" int yywrap (void);
extern char yytext[];
extern int yyleng;

using namespace std;

#define YYDEBUG 1

%}

/* Declare the type of values in the grammar */

%union{ 
  long      ival;           /* Long value           */
  double    dval;           /* Double value         */
  char      cval;           /* Char value           */
  char*     strval;         /* Char * value         */
}

/*
 * Token types: These are returned by the lexer
 */

%token <strval> SXF_IDENTIFIER
%token <cval>   SXF_CHARACTER_LITERAL
%token <dval>   SXF_NUMERIC_LITERAL

%token 		SXF_ACCELERATOR
%token 		SXF_SEQUENCE
%token 		SXF_ENDSEQUENCE
%token 		SXF_DESIGN
%token 		SXF_BODY
%token 		SXF_BODY_DEV
%token 		SXF_AT
%token 		SXF_L
%token 		SXF_ARC
%token 		SXF_N


/*
 * Production starts here
 * These rules have been derived from the Accelerator Description
 * eXchange Format (ADXF) to fit the SXF specification.
 */

%%

start				: sequence;


/* Sequence */

sequence			: SXF_IDENTIFIER SXF_SEQUENCE 
				{ 
				  if(!SXF::AcceleratorReader::s_reader->open($1)) 
					SXF::AcceleratorReader::s_reader->echo().cfe_error()
				    		<< "\n*** CFE Error in the sequence declaration" << endl;  
                                }
				'{' sequence_nodes sequence_end '}'
				{ SXF::AcceleratorReader::s_reader->close(); }
				;

sequence_nodes			: /* empty */
				| sequence_nodes sequence_node
				;

sequence_node			: element  ';'
				| error
				{
				   SXF::AcceleratorReader::s_reader->echo().syntax_error() 
                                     << "\n*** Syntax Error: in the seq. node declaration" << endl;
 			        }
				';'
				{ 
				   yyerrok; 
                                }
				; 

sequence_end			: SXF_ENDSEQUENCE sequence_end_attributes
				;

sequence_end_attributes		: /* empty */ 
				| sequence_end_attributes sequence_end_attribute
				;

sequence_end_attribute		: SXF_AT '=' SXF_NUMERIC_LITERAL 
				{ SXF::AcceleratorReader::s_reader->getSequence()->setLength($3);}
				;

/* Element */

element				: elem_key_and_type 
				'{' elem_header elem_buckets '}'
				{ SXF::AcceleratorReader::s_reader->closeElement(); }
				;
			
elem_key_and_type		: SXF_IDENTIFIER SXF_IDENTIFIER 
				{ 
				  if(!SXF::AcceleratorReader::s_reader->openElement($2, $1)) 
				    SXF::AcceleratorReader::s_reader->echo().cfe_error()
				       << "\n*** CFE Error in the element declaration" << endl;  
                                }
				| SXF_IDENTIFIER 
				{ 
				  if(!SXF::AcceleratorReader::s_reader->openElement($1, "")) 
				    SXF::AcceleratorReader::s_reader->echo().cfe_error()
				      << "\n*** CFE Error in the element declaration" << endl;
                                }
				;

/* Element Header */

elem_header			: /* empty */
				| elem_header elem_header_attribute
				;

elem_header_attribute		: SXF_DESIGN '=' SXF_IDENTIFIER
				{ SXF::AcceleratorReader::s_reader->getElement()->setDesign($3);}
				| SXF_L '='      SXF_NUMERIC_LITERAL
				{ SXF::AcceleratorReader::s_reader->getElement()->setLength($3);}
				| SXF_ARC '='    SXF_NUMERIC_LITERAL
				{ SXF::AcceleratorReader::s_reader->getElement()->setLength($3);}
				| SXF_AT '='     SXF_NUMERIC_LITERAL
				{ SXF::AcceleratorReader::s_reader->getElement()->setAt($3);}
				| SXF_N '='      SXF_NUMERIC_LITERAL
				{ SXF::AcceleratorReader::s_reader->getElement()->setN($3);}
				;

/* Element Buckets */

elem_buckets			: elem_bodies
				| elem_buckets elem_bucket 
				;

/* Element Body */

elem_bodies		        : /* empty */
				| elem_bodies elem_body
				;

elem_body			: elem_body_key '{' elem_attributes '}'
				{ SXF::AcceleratorReader::s_reader->getElement()->closeBucket();}			
				;

elem_body_key			: SXF_BODY '='
				{ 
                                  if(!SXF::AcceleratorReader::s_reader->getElement()->openBody("body")) 
				    SXF::AcceleratorReader::s_reader->echo().cfe_error()
				      << "\n*** CFE Error in the element body declaration" << endl;
                                }
				| SXF_BODY_DEV '='
				{ 
                                  if(!SXF::AcceleratorReader::s_reader->getElement()->openBody("body.dev")) 
				    SXF::AcceleratorReader::s_reader->echo().cfe_error()
				      << "\n*** CFE Error in the element body.dev declaration" << endl;
                                }
				;

/* Element Bucket */

elem_bucket			: elem_bucket_key '{' elem_attributes '}'
				{ SXF::AcceleratorReader::s_reader->getElement()->closeBucket();}
				;

elem_bucket_key			: SXF_IDENTIFIER '='
				{ 
				  if(!SXF::AcceleratorReader::s_reader->getElement()->openBucket($1)) 
				   SXF::AcceleratorReader::s_reader->echo().cfe_error()
				     << "\n*** CFE Error in the element attribute or bucket(" << $1 
				     << ") declaration" << endl;	
                                }
				;

/* Element Attribute */

elem_attributes			: /* empty */
				| elem_attributes elem_attribute 
				;

elem_attribute			: elem_attribute_key elem_attribute_value 
				{ SXF::AcceleratorReader::s_reader->getElement()->getBucket()->closeAttribute(); }
				;


elem_attribute_key		: SXF_IDENTIFIER  '='  
				{ 
                                  if(!SXF::AcceleratorReader::s_reader->getElement()->getBucket()->openAttribute($1))
                                    SXF::AcceleratorReader::s_reader->echo().cfe_error() 
				     << "\n*** CFE Error in the element attribute(" << $1 
				     << ") declaration" << endl;	
                                }
				;

/* Element Attribute Values */

elem_attribute_value		: elem_attribute_scalar 
				| elem_attribute_array 
				| elem_attribute_hash
				;

elem_attribute_scalar		: SXF_NUMERIC_LITERAL
				{ 
                                  if(!SXF::AcceleratorReader::s_reader->getElement()->getBucket()->setScalarValue($1))
				    SXF::AcceleratorReader::s_reader->echo().cfe_error()
				     << "\n*** CFE Error in setting scalar value(" 
				     << $1 << ")" << endl;
                                }
                       		;

elem_attribute_array		: 
				'['
				{ 
                                  if(!SXF::AcceleratorReader::s_reader->getElement()->getBucket()->openArray())
				    SXF::AcceleratorReader::s_reader->echo().cfe_error()
				     << "\n*** CFE Error in opening an array" << endl;
                                }
				elem_attribute_avalues 
				']'
				{ SXF::AcceleratorReader::s_reader->getElement()->getBucket()->closeArray();}	
				;

elem_attribute_hash 		: 
				'{' 
				{ 
                                  if(!SXF::AcceleratorReader::s_reader->getElement()->getBucket()->openHash())
				    SXF::AcceleratorReader::s_reader->echo().cfe_error()
				     << "\n*** CFE Error in opening a hash" << endl;
                                }
				elem_attribute_hvalues 
				'}'
				{ SXF::AcceleratorReader::s_reader->getElement()->getBucket()->closeHash();}
				;

elem_attribute_avalues		: /* empty */
				| elem_attribute_avalues elem_attribute_avalue
				;	

elem_attribute_avalue		: SXF_NUMERIC_LITERAL
				{ SXF::AcceleratorReader::s_reader->getElement()->getBucket()->setArrayValue($1);}
                       		;


elem_attribute_hvalues		: /* empty */
				| elem_attribute_hvalues  elem_attribute_hvalue
				;

elem_attribute_hvalue		: '[' SXF_NUMERIC_LITERAL ']' '=' SXF_NUMERIC_LITERAL
				{ 
                                   if(!SXF::AcceleratorReader::s_reader->getElement()->getBucket()->setHashValue($5, (int)$2))
					SXF::AcceleratorReader::s_reader->echo().cfe_error()
					<< "\n*** CFE Error in setting a hash value( ["
					<< $2 << "] = " << $5 << ")" <<  endl;
                                }
				; 	
	
%%

/* programs */

int yywrap() { 
  SXF::AcceleratorReader::s_reader->echo().write_status();
  return 1; 
}

/* Report an error situation discovered in a production */
void yyerror(char* ) {
   SXF::AcceleratorReader::s_reader->echo().syntax_error() << "\n*** Syntax Error: in the last lines" << endl;	
}

