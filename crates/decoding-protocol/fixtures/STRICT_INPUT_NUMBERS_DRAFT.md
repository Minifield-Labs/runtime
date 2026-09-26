# Strict input admission and numeric validation, draft 1

This addendum is normative for decoding-protocol draft5. It clarifies ingestion separately from runtime generated-value grammar; structural draft4 schema planning is retained. It is a proposed experiment contract pending Python/Rust parity, not a released backend contract.

## Strict documents and bounded parsing

Catalog/schema, event, annotation-derived teacher, and trace fixture JSON documents must be parsed with decoded duplicate-key rejection before map insertion, preservation of object insertion order, strict UTF8 and paired-surrogate validation, and rejection of non-JSON NaN/Infinity. Thus keys `a` and `\u0061` collide. Duplicate checks apply to data nested inside const/enum/default/examples too; those data objects are not traversed as schema keywords. Leading/trailing/inter-token whitespace is allowed in source JSON. The separate runtime generated-value grammar still forbids whitespace outside strings.

Retain each numeric token's raw decimal spelling through admission. Do not feed an already lossy float-only parse into checks that need the spelling. The parser may expose a typed tree plus a path-to-raw-number sidecar or a number wrapper. Public traces contain no extra raw-number metadata; admitted semantic numbers serialize through the existing safe JCS operation. Programmatic inputs lacking raw spellings use their valid canonical JSON serialization as the admission spelling, and reject unsupported numeric Python/Rust types rather than applying implicit coercions.

Initial source-document limits are 4 MiB UTF8 bytes, nesting depth64 (root depth0),100000 value nodes (root and each object value/array item count, keys do not), and1024 ASCII bytes per number token. Check limits before unbounded allocation. Resolved-schema expansion separately has depth64 and100000 nodes/work visits; cycle detection remains mandatory. Existing model context and generated-argument budgets apply additionally. Exponent handling must classify huge exponents without allocating a huge power of ten; the token-byte cap is not permission to allocate 10**an_arbitrary_exponent.

## One admitted numeric meaning

Raw lexemes support admission checks; they do not create an undisclosed arbitrary-precision number type on the wire. Apply draft4 nearest/ties-even finite binary64 conversion, nonzero-underflow rejection, and safe integral-domain guards to numeric schema values and data values alike. Schema constraints, bounds and numeric const/enum semantic equality refer to these admitted converted values. Booleans never compare equal to numbers. Preserve negative-zero token history, but typed/canonical assembly yields zero.

Accordingly source schema `{"minimum":1.0000000000000001}` has converted minimum1. Its original bytes/hash are preserved as provenance; its declared numeric interpretation is binary64. The implementation must record this policy, not claim arbitrary-precision validation of the raw source decimal. Likewise numeric const1 and const1.0000000000000001 denote the same admitted value. For an integer-typed runtime value, raw1.0000000000000001 still fails the additional raw-decimal integrality requirement even though it converts to1. That lexical rule is extra to JSON Schema's normal typed-value validation.

Numeric schema keywords that require nonnegative integer values (minLength/maxLength/minItems/maxItems) must satisfy raw mathematical integrality, nonnegativity and safe-integer guards before conversion. Do not let rounding turn an invalid keyword into a valid integer. Numeric bounds require finite admitted numbers. multipleOf requires a strictly positive admitted number; zero/negative values are invalid schemas. JSON Schema type/keyword metavalidation still applies.

## Exact decimal assertions over canonical wire numbers

Once admitted, define Q(x) as the exact rational represented by the RFC8785 canonical decimal spelling of binary64 x. This is the decimal meaning exposed on the JSON wire. Bounds compare Q(instance) and Q(bound). Numeric const/enum/uniqueItems equality compares admitted numeric values, equivalent to Q equality. multipleOf passes exactly when Q(instance)/Q(divisor) is an integer. Use bounded exact decimal coefficient/exponent or rational arithmetic; never an epsilon, rounded float quotient, or an approximate remainder.

Examples:0.3 is a multiple of0.1;0.30000000000000004 is not.0.07 is a multiple of0.01. Binary64-converted source1.0000000000000001 canonicalizes to1 and equals const1. Source value9007199254740992 fails the safe-domain guard before any schema checks. Source5e-324 is admitted as a finite subnormal;1e-400 fails nonzero-underflow admission. Exact raw mathematical integrality can be checked from sign/coefficient/fraction/exponent without constructing an unbounded integer.

This explicitly resolves the vague draft4 phrase original validator semantics: validate the original schema's full structure under this declared numeric interpretation; do not inherit a library's accidental floating quotient behavior. Preserve unmodified schema bytes and also hash the interpreted canonical schema. Full original structure includes all unselected union branches, sibling constraints and object closure rules; internal schema-plan choice does not replace it.

## Independent validation and compatibility

The independent Python Draft202012Validator oracle must consume the admitted canonical JSON schema and instance with integers parsed as int and fractional/exponent numbers as exact Decimal. This exercises the library's original assertion/applicator implementations with decimal inputs; do not replace multipleOf with the implementation under test. Ensure integer checking regards mathematically integral numeric values correctly: parse integral canonical decimals/exponents as int, not Decimal, when feeding this oracle. Guard booleans separately. Record Python/jsonschema/Decimal context settings and demonstrate that every arithmetic operation needed for bounded canonical binary64 numbers is exact (use a local context precision of at least2048; no global context mutation). Tests compare this independent validator with the Rust constraint implementation.

Also run the supplied package/backend validator using its original pinned environment against the actual admitted corpus, preserving its historical behavior. Report any disagreement between the declared protocol oracle and actual backend as a compatibility failure that prevents installation/action execution for the affected schema/value. Do not silently choose whichever validator accepts, discard the schema, rewrite an original report, or weaken a rule. Agreement on the current corpus does not assert equivalence for all possible future JSON schemas.

Node RegExp fixtures record the exact Node/V8 versions, pattern string, flags (empty unless a separately admitted extension declares otherwise), decoded test string and Boolean outcome. ECMAScript search semantics, Unicode scalar length, and JSON source parsing are separate checks. Keep paired-surrogate/non-BMP/final-newline fixtures. The13 actual corpus pattern forms must all agree with Rust before admission.

## Cross-language fixtures

Include duplicate literal/escaped keys in top-level and nested schema data, source order and supplementary Unicode keys, safe-integer boundaries, underflow/subnormal/overflow, negative zero, huge exponent tokens, raw-integral versus rounded-integral values, const/enum duplicate numeric equality, integer keyword lexical rejection, decimal multipleOf examples above, and canonical float formatting boundaries. Verify source-byte budgets, depth/node limits, and exact limit plus one failures without excessive allocation. A caller that only has a float-parsed source cannot claim to have performed raw lexical admission.

## Primary references

The input and serialization choices follow [RFC8785 sections3.1 and3.2.2.3](https://www.rfc-editor.org/rfc/rfc8785). Decimal division follows the mathematical assertion in [JSON Schema2020-12 validation section6.2.1](https://json-schema.org/draft/2020-12/json-schema-validation#section-6.2.1). The binary64 domain, raw-integer admission rule and exact canonical-decimal interpretation are explicit choices of this application protocol.
