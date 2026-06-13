import { marked } from "marked";
import markedKatex from "marked-katex-extension";
import sanitizeHtml from "sanitize-html";

marked.use(
  markedKatex({
    throwOnError: false,
    nonStandard: true,
  }),
);

marked.setOptions({
  gfm: true,
  breaks: false,
});

const DISPLAY_MATH_RE = /\$\$[\s\S]*?\$\$/g;
const INLINE_MATH_RE = /(?<!\$)\$(?!\$)(?:\\.|[^$\n\\]|\\[^$])+?\$(?!\$)/g;

const SENTENCE_WORD =
  /^(?:The|But|Hence|However|Therefore|Thus|For|Using|Consider|And|Or|Is|Are|Was|In|On|To|Not|Also|This|That|With|From|Let|We|It|An|At|By|As|If|So|Do|Be|A|Of|However|norm|converges|uniformly|polynomial|complete|series)$/i;

function normalizeLatexDelimiters(markdown: string): string {
  return markdown
    .replace(/\\\[([\s\S]*?)\\\]/g, (_, inner: string) => `$$\n${inner.trim()}\n$$`)
    .replace(/\\\(([\s\S]*?)\\\)/g, (_, inner: string) => `$${inner.trim()}$`);
}

function wrapLatexEnvironments(text: string): string {
  return text.replace(/\\begin\{([^}]+)\}([\s\S]*?)\\end\{\1\}/g, (_, env: string, body: string) => {
    return `$$\n\\begin{${env}}${body}\\end{${env}}\n$$`;
  });
}

function wrapLeftRightMath(text: string): string {
  return text.replace(
    /\\left(?:\||\(|\[)[\s\S]*?\\right(?:\||\)|\])(?:[_^](?:\{[^{}]*\}|[A-Za-z0-9]+))?/g,
    (block) => `$${block.trim()}$`,
  );
}

function wrapNorms(text: string): string {
  return text.replace(/\\?\|([^|]+)\|(?:_\{[^{}]*\}|_[A-Za-z0-9^{}]+)?/g, (block, inner: string) => {
    if (inner.includes("\\")) return block;
    return `$${block.trim()}$`;
  });
}

function wrapEquations(text: string): string {
  return text.replace(
    /(?:^|[\s(])([A-Za-z][A-Za-z0-9_^\-]*(?:\([^)]*\))?=[^.;]{2,}?)(?=\s+(?:For|Using|But|The|Hence|However|Thus|and|is)\b|[.;]|$)/g,
    (full, expr: string) => {
      if (!expr.includes("\\") && !/[_^]/.test(expr)) return full;
      const prefix = full.slice(0, full.indexOf(expr));
      return `${prefix}$${expr.trim()}$`;
    },
  );
}

function splitOutsideProtectedMath(text: string): string[] {
  const protectedRe = new RegExp(
    `${DISPLAY_MATH_RE.source}|${INLINE_MATH_RE.source}`,
    "g",
  );
  const parts: string[] = [];
  let last = 0;
  for (const match of text.matchAll(protectedRe)) {
    const index = match.index ?? 0;
    if (index > last) parts.push(text.slice(last, index));
    parts.push(match[0]);
    last = index + match[0].length;
  }
  parts.push(text.slice(last));
  return parts;
}

function isMathChar(ch: string): boolean {
  return /[\\{}_^=+\-*/(),.;|0-9a-zA-Z]/.test(ch);
}

function findBackslashSpanEnd(segment: string, start: number): number {
  let i = start;
  let braceDepth = 0;

  while (i < segment.length) {
    const ch = segment[i];

    if (ch === "{") braceDepth++;
    else if (ch === "}" && braceDepth > 0) braceDepth--;

    if (braceDepth === 0 && i > start && (ch === " " || ch === "\t")) {
      const rest = segment.slice(i).trimStart();
      const word = rest.match(/^([A-Za-z]+)/);
      if (word && SENTENCE_WORD.test(word[1])) return i;
    }

    if (braceDepth === 0 && i > start && (ch === "." || ch === ",")) {
      const after = segment.slice(i + 1).trimStart();
      if (after && /^[A-Z]/.test(after)) return i;
    }

    if (braceDepth === 0 && i > start && !isMathChar(ch) && ch !== " ") return i;

    i++;
    if (i - start > 420) break;
  }

  return i;
}

function wrapBackslashSpans(segment: string): string {
  if (!segment.includes("\\")) return segment;

  const out: string[] = [];
  let cursor = 0;

  while (cursor < segment.length) {
    const start = segment.indexOf("\\", cursor);
    if (start === -1) {
      out.push(segment.slice(cursor));
      break;
    }

    out.push(segment.slice(cursor, start));
    const end = findBackslashSpanEnd(segment, start);
    const expr = segment.slice(start, end).trim();

    if (expr.length > 1) out.push(`$${expr}$`);
    else out.push(expr);

    cursor = end;
  }

  return out.join("");
}

function protectPass(text: string, fn: (chunk: string) => string): string {
  return splitOutsideProtectedMath(text)
    .map((part) => {
      if (part.startsWith("$$") || (part.startsWith("$") && part.endsWith("$"))) return part;
      return fn(part);
    })
    .join("");
}

function collapseBrokenDelimiters(text: string): string {
  return text.replace(/\${3,}/g, "$$").replace(/\$\$\s*\$(?!\$)/g, "$$").replace(/(?<!\$)\$\s*\$\$/g, "$$");
}

function cleanupChunkMarkdown(text: string): string {
  return text
    .replace(/^>\s+/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function prepareChunkMarkdown(text: string): string {
  let md = cleanupChunkMarkdown(text);
  md = normalizeLatexDelimiters(md);
  md = protectPass(md, wrapLatexEnvironments);
  md = protectPass(md, wrapLeftRightMath);
  md = protectPass(md, wrapNorms);
  md = protectPass(md, wrapEquations);
  md = protectPass(md, wrapBackslashSpans);
  md = collapseBrokenDelimiters(md);
  return md;
}

export function renderChunkMarkdown(text: string): string {
  const rawHtml = marked.parse(prepareChunkMarkdown(text)) as string;
  return sanitizeHtml(rawHtml, {
    allowedTags: [
      ...sanitizeHtml.defaults.allowedTags,
      "h1",
      "h2",
      "h3",
      "span",
      "math",
      "annotation",
      "semantics",
      "mrow",
      "mi",
      "mn",
      "mo",
      "msup",
      "msub",
      "mfrac",
      "msqrt",
      "mspace",
      "mtext",
      "mtable",
      "mtr",
      "mtd",
    ],
    allowedAttributes: {
      ...sanitizeHtml.defaults.allowedAttributes,
      span: ["class", "style"],
      div: ["class"],
      code: ["class"],
      a: ["href", "name", "target", "rel"],
      math: ["xmlns", "display"],
      annotation: ["encoding"],
      semantics: [],
      mrow: [],
      mi: [],
      mn: [],
      mo: [],
      msup: [],
      msub: [],
      mfrac: [],
      msqrt: [],
      mspace: [],
      mtext: [],
      mtable: [],
      mtr: [],
      mtd: [],
    },
  });
}
