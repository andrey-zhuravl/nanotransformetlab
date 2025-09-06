parser grammar AntlrFlowParser;

options { tokenVocab = AntlrFlowLexer; }

// SECTION: general

script
    : NL* packageHeader importList (statement semi)* statement? EOF
    ;

repl
    : NL* importList (statement semi)* statement? EOF
    ;

packageHeader
    : (PACKAGE identifier semi?)?
    ;

importList
    : importHeader*
    ;

importHeader
    : IMPORT (identifier COLON)? identifier (DOT identifier)* (DOT MULT | importAlias)? semi?
    ;

importAlias
    : AS simpleIdentifier
    ;


declaration
    : classDeclaration
    | functionDeclaration
    | propertyDeclaration
    ;

// SECTION: classes

classDeclaration
    : modifiers? CLASS NL* simpleIdentifier
      (NL* COLON NL* delegationSpecifiers)?
      (NL* classBody)?
    ;


classBody
    : LBRACE NL* classMemberDeclarations NL* RBRACE
    ;


delegationSpecifiers
    : annotatedDelegationSpecifier (NL* COMMA NL* annotatedDelegationSpecifier)*
    ;

delegationSpecifier
    : userType
    ;

annotatedDelegationSpecifier
    : NL* delegationSpecifier
    ;


typeParameters
    : NL* typeParameter NL*
    ;

typeParameter
    : simpleIdentifier
    ;

// SECTION: classMembers

classMemberDeclarations
    : (classMemberDeclaration semis?)*
    ;

classMemberDeclaration
    : declaration
    ;


functionValueParameters
    : LPAREN NL*
      (functionValueParameter
      (NL* COMMA NL*
      functionValueParameter)*)?
      RPAREN
    ;

functionValueParameter
    : parameter (NL* ASSIGN NL* expression)?
    ;

functionDeclaration
    : modifiers?
      FUN NL* simpleIdentifier
      NL* functionValueParameters
      (NL* COLON NL* type)?
      (NL* functionBody)?
    ;

functionBody
    : block
    | ASSIGN NL* expression
    ;

variableDeclaration
    : NL* simpleIdentifier (NL* COLON NL* type)?
    ;

multiVariableDeclaration
    : LPAREN NL* variableDeclaration (NL* COMMA NL* variableDeclaration)* (NL* COMMA)? NL* RPAREN
    ;

propertyDeclaration
    : modifiers? (VAL | VAR)
      (NL* typeParameters)?
      (NL* variableDeclaration)
      (NL* ASSIGN NL* expression)?
      (NL* SEMICOLON)? NL*
    ;

parameter
    : simpleIdentifier NL* COLON NL* type
    ;


// SECTION: types

type
    : (parenthesizedType | nullableType | typeReference)
    ;

typeReference
    : userType
    ;

nullableType
    : (typeReference | parenthesizedType) NL* quest+
    ;

quest
    : QUESTION
    | QUESTION_WS
    ;

userType
    : simpleUserType (NL* DOT NL* simpleUserType)*
    ;

simpleUserType
    : simpleIdentifier (NL* typeArguments)? # declaredType
    ;

typeProjection
    : type
    ;


parenthesizedType
    : LPAREN NL* type NL* RPAREN
    ;


// SECTION: statements

statements
    : (statement (semis statement)*)? semis?
    ;

statement
    : (declaration | assignment | expression)
    ;

controlStructureBody
    : block
    | statement
    ;

block
    : LBRACE NL* statements NL* RBRACE
    ;

assignment
    : (directlyAssignableExpression ASSIGN | assignableExpression assignmentAndOperator) NL* expression
    ;

semi
    : (SEMICOLON | NL) NL*
    ;

semis
    : (SEMICOLON | NL)+
    ;

// SECTION: expressions

expression
    : disjunction
    ;

disjunction
    : conjunction (NL* OR NL* conjunction)*
    ;

conjunction
    : equality (NL* AND NL* equality)*
    ;

equality
    : comparison (equalityOperator NL* comparison)*
    ;

comparison
    : genericCallLikeComparison (comparisonOperator NL* genericCallLikeComparison)*
    ;

genericCallLikeComparison
    : infixOperation callSuffix*
    ;

infixOperation
    : elvisExpression (inOperator NL* elvisExpression | isOperator NL* type)*
    ;

elvisExpression
    : infixFunctionCall (NL* elvis NL* infixFunctionCall)*
    ;

elvis
    : QUESTION COLON
    ;

infixFunctionCall
    : rangeExpression (simpleIdentifier NL* rangeExpression)*
    ;

rangeExpression
    : additiveExpression ((RANGE | RANGE_UNTIL) NL* additiveExpression)*
    ;

additiveExpression
    : multiplicativeExpression (additiveOperator NL* multiplicativeExpression)*
    ;

multiplicativeExpression
    : asExpression (multiplicativeOperator NL* asExpression)*
    ;

asExpression
    : prefixUnaryExpression (NL* asOperator NL* type)*
    ;

prefixUnaryExpression
    : NL* unaryPrefix* postfixUnaryExpression
    ;

unaryPrefix
    : prefixUnaryOperator NL*
    ;

postfixUnaryExpression
    : primaryExpression postfixUnarySuffix*
    ;

postfixUnarySuffix
    : postfixUnaryOperator
    | callSuffix
    | navigationSuffix
    ;

directlyAssignableExpression
    : simpleIdentifier
    ;

assignableExpression
    : prefixUnaryExpression
    | parenthesizedAssignableExpression
    ;

parenthesizedAssignableExpression
    : LPAREN NL* assignableExpression NL* RPAREN
    ;

navigationSuffix
    : memberAccessOperator NL* (simpleIdentifier | parenthesizedExpression | CLASS)
    ;

callSuffix
    : valueArguments
    ;

typeArguments
    : LT NL* typeProjection (NL* COMMA NL* typeProjection)* (NL* COMMA)? NL* GT
    ;

valueArguments
    : LPAREN NL* (valueArgument (NL* COMMA NL* valueArgument)* (NL* COMMA)? NL*)? RPAREN
    ;

valueArgument
    : NL* (simpleIdentifier NL* ASSIGN NL*)? MULT? NL* expression
    ;

primaryExpression
    : parenthesizedExpression
    | simpleIdentifier
    | literalConstant
    | stringLiteral
    | propertyDeclaration
    | ifExpression
    | jumpExpression
    ;

parenthesizedExpression
    : LPAREN NL* expression NL* RPAREN
    ;


literalConstant
    : BooleanLiteral
    | IntegerLiteral
    | HexLiteral
    | BinLiteral
    | CharacterLiteral
    | RealLiteral
    | NullLiteral
    | LongLiteral
    | UnsignedLiteral
    ;

stringLiteral
    : lineStringLiteral
    ;

lineStringLiteral
    : QUOTE_OPEN lineStringPart* QUOTE_CLOSE
    ;

lineStringPart
    : (lineStringContent | lineStringExpression)
    ;

lineStringContent
    : LineStrText
    | LineStrEscapedChar
    ;

lineStringExpression
    : LineStrExprStart NL* expression NL* RBRACE
    ;


ifExpression
  : IF NL* LPAREN NL* expression NL* RPAREN NL*
      (
         controlStructureBody
       | controlStructureBody? NL* SEMICOLON? NL* ELSE NL* ( controlStructureBody | SEMICOLON )
       | SEMICOLON
      )
  ;

jumpExpression
    : RETURN expression?
    ;

assignmentAndOperator
    : ADD_ASSIGN
    | SUB_ASSIGN
    | MULT_ASSIGN
    | DIV_ASSIGN
    | MOD_ASSIGN
    ;

equalityOperator
    : NOT_EQUAL
    | EQUAL
    ;

comparisonOperator
    : LT
    | GT
    | LTE
    | GTE
    ;

inOperator
    : IN
    | NOT_IN
    ;

isOperator
    : IS
    | NOT_IS
    ;

additiveOperator
    : PLUS
    | MINUS
    ;

multiplicativeOperator
    : MULT
    | DIV
    | MOD
    ;

asOperator
    : AS
    ;

prefixUnaryOperator
    : INCR
    | DECR
    | MINUS
    | PLUS
    | excl
    ;

postfixUnaryOperator
    : INCR
    | DECR
    | NOT_WS excl
    ;

excl
    : NOT_WS
    | NOT
    ;

memberAccessOperator
    : NL* DOT
    ;

// SECTION: modifiers

modifiers
    : (modifier)+
    ;

modifier
    : (visibilityModifier
    | functionModifier
    | propertyModifier
    | inheritanceModifier) NL*
    ;


visibilityModifier
    : PUBLIC
    | PRIVATE
    | PROTECTED
    ;


functionModifier
    : EXTERNAL
    ;

propertyModifier
    : CONST
    ;

inheritanceModifier
    : ABSTRACT
    | OPEN
    ;


// SECTION: identifiers
simpleIdentifier
    : Identifier
    ;


identifier
    : simpleIdentifier (NL* DOT simpleIdentifier)*
    ;