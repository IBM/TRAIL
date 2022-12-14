
/*
   Based on http://www.cyc.com/wp-content/uploads/2015/07/CycLsyntax.pdf
   and contents of OpenCyc
 */

grammar CycL;


constant
	:	NAMECHARSEQUENCE
	|   QUOTEDCHARSEQUENCE//VERONIKA don't kno why this is necessary and not covered by below
	|   QUOTEDSTRING
	;


variable
	:	VARCHARSEQUENCE
	;


cyclterm //set denoting functions where we need extra processing
	:	thesetof
	|   kappa
	;


term
	:	variable
	|   constant
	|	OPEN operator arguments CLOSE
	|   cyclterm
	|   reserved
	;


reserved
	:   EQUALS
    |   NOT
    |   AND
    |   OR
    |   IF
    |   ONLY_IF
    |   IFF
    |   EXISTS
    |   FORALL
    |   CYC_MT
    |   CYC_COMMENT
    |   CYC_THESETOF
    |   CYC_PRETTYSTR
    |   CYC_PRETTYSTRCANONICAL
    |   CYC_KAPPA
    |   CYC_TRUERULE
    |   CYC_HOLDSIN
    |   CYC_IST
    |   CYC_EXCEPT
    |   CYC_RELATIONALLINSTANCE
    |   CYC_RELATIONALLEXISTS
	;


operator
	:	term
	;


arguments
	:	term*
	;


cyclstatement
	:	microtheory
	|   comment
	|   prettystring
	|   excepts
	;


microtheory
	:	OPEN 'in-microtheory' constant CLOSE
	;


comment
	:	OPEN 'comment' ( NAMECHARSEQUENCE | reserved | sentence) QUOTEDSTRING CLOSE
	;


prettystring
    :   OPEN CYC_PRETTYSTR term (QUOTEDCHARSEQUENCE | QUOTEDSTRING) CLOSE
    |   OPEN CYC_PRETTYSTR term OPEN 'UnicodeStringFn' (QUOTEDCHARSEQUENCE | QUOTEDSTRING) CLOSE CLOSE
    |   OPEN 'prettyString-Canonical' term (QUOTEDCHARSEQUENCE | QUOTEDSTRING) CLOSE
    ;


kappa
    :   OPEN 'Kappa' OPEN variable* CLOSE sentence CLOSE
    ;


truerule
    :   OPEN 'trueRule' sentence sentence CLOSE
    ;


holdsin
    :   OPEN 'holdsIn' term sentence CLOSE
    ;


ist
    :   OPEN 'ist' term sentence CLOSE
    ;


relationallexists
    :   OPEN 'relationAllExists' predicate term term CLOSE
    ;


relationallinstance
    :   OPEN 'relationAllInstance' predicate term term CLOSE
    ;


thesetof
    :   OPEN 'TheSetOf' variable sentence CLOSE
    ;


excepts
    :   OPEN 'except' sentence CLOSE
    ;


sentence
	:	atomsent
	|	boolsent
	|	quantsent
	|   truerule
	|   relationallinstance
	|   relationallexists
	|   holdsin
	|   ist
	|   variable//in cycl we may have (implies ?X ?Y)
	;


atomsent
	:	equation
	|	atom
	;


equation
	:	OPEN 'equals' term term CLOSE
	;

	
atom
	:	OPEN predicate arguments CLOSE
	; 


predicate
	:	term
	;


boolsent
	:	OPEN 'not' sentence CLOSE
	|   OPEN ('and' | 'or') sentence* CLOSE
	|	OPEN ('implies' | '<==' | 'equiv') sentence sentence CLOSE
	;


quantsent
	:	OPEN ('thereExists' | 'forAll') variable sentence CLOSE
	;


statement
	:   cyclstatement
	|   sentence
	;


theory
	:	statement+
	;


// Delimiters
OPEN			: '(';
CLOSE			: ')';
STRINGQUOTE	: '"';
BACKSLASH	: '\\';


// Characters
fragment
CHAR			: [0-9~!#$%^&*@_+{}|:<>`'\-=[\];,./A-Za-z];

fragment
DIGIT			: [0-9];

fragment
HEXA			: [0-9A-Fa-f];


// Quoting within strings
fragment
NONASCII		
	: '\\' 'u' HEXA HEXA HEXA HEXA 
	| '\\' 'U' HEXA HEXA HEXA HEXA HEXA HEXA
	| '&' 'u' HEXA+
	;

fragment
INNERSTRINGQUOTE		: '\\"' ;

fragment
INNERBACKSLASH		: '\\\\';

//NUMERAL				: DIGIT+;

QUOTEDCHARSEQUENCE : STRINGQUOTE CHAR+ STRINGQUOTE ;

QUOTEDSTRING : STRINGQUOTE (WHITE | OPEN | CLOSE | CHAR | '?' | NONASCII | INNERSTRINGQUOTE | INNERBACKSLASH )* STRINGQUOTE ;


//Reserved tokens
EQUALS					:	'equals';
NOT					:	'not';
AND					:	'and';
OR						:	'or';
IF						:	'implies';
ONLY_IF						:	'<==';
IFF						:	'equiv';
EXISTS					:	'thereExists';
FORALL				:	'forAll';
CYC_MT			:	'in-microtheory';
CYC_COMMENT			:	'comment';
CYC_THESETOF    :	'TheSetOf';
CYC_PRETTYSTR    :	'prettyString';
CYC_PRETTYSTRCANONICAL    :	'prettyString-Canonical';
CYC_KAPPA    :	'Kappa';
CYC_TRUERULE    :	'trueRule';
CYC_HOLDSIN    :	'holdsIn';
CYC_IST    :	'ist';
CYC_EXCEPT    :	'except' ;
CYC_RELATIONALLINSTANCE    :	'relationAllInstance';
CYC_RELATIONALLEXISTS    :	'relationAllExists';
//CYC_ISA			:	'isa';
//CYC_GENL			:	'genl';
//CYC_ARGISA			:	'argisa';
//CYC_ARGGENL			:	'argGenl';
//VERONIKA: here we could add genls, argformat etc. ie special cycl keys that can be used to optimize reasoning and make it consistent (eg using argformat etc)
//problem if we use isa, genls, etc here, we have complex predicates, eg (#$isa ?APPLE (#$FruitFn #$AppleTree))
//TODO continue


//RESERVED_TOKEN
//    :   EQUALS
//    |   NOT
//    |   AND
//    |   OR
//    |   IF
//    |   ONLY_IF
//    |   IFF
//    |   EXISTS
//    |   FORALL
//    |   CYC_MT
//    |   CYC_COMMENT
//    |   CYC_THESETOF
//    |   CYC_PRETTYSTR
//    |   CYC_PRETTYSTRCANONICAL
//    |   CYC_KAPPA
//    |   CYC_TRUERULE
//    |   CYC_HOLDSIN
//    |   CYC_IST
//    |   CYC_EXCEPT
//    |   CYC_RELATIONALLINSTANCE
//    |   CYC_RELATIONALLEXISTS
//    ;


NAMECHARSEQUENCE
	:	CHAR+ //( CHAR  (CHAR | STRINGQUOTE | NAMEQUOTE | BACKSLASH)* )
//	|   STRINGQUOTE CHAR+ STRINGQUOTE
	;


VARCHARSEQUENCE
	:	( '?'  (CHAR)* )
	;
//
//
//TEXT
//    : (CHAR | '?' | QUOTE )+//| OPEN | CLOSE
//    ;


WHITE
	:	[ \t\n\r\u000B]							-> skip
	;


// VERONIKA: not sure if these exist in NextKB
//BLOCKCOMMENT
//	:	'/*' (BLOCKCOMMENT | .)*? '*/' 	->	skip // nesting allowed (but should it be?)
//	;


//Prettystring1
//    :   OPEN 'prettyString' CHARSEQUENCE (QUOTEDCHARSEQUENCE | QUOTEDSTRING) CLOSE          ->	skip
//    ;
//
//Prettystring2
//    :   OPEN 'prettyString' CHARSEQUENCE OPEN 'UnicodeStringFn' (QUOTEDCHARSEQUENCE | QUOTEDSTRING) CLOSE CLOSE         ->	skip
//    ;

LineComment
	:	'//' ~[\u000A\u000D]*			->	skip
	;

CycLineComment
	:	';' ~[\u000A\u000D]*			->	skip
	;

CycLineComment1
	:	';'+[\u000A\u000D]*			->	skip//need this one if line break directly after ;
	;
