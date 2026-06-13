import { describe, expect, test } from "bun:test";
import { prepareChunkMarkdown } from "./chunk-markdown.js";

const HILBERT_CHUNK = `## Proof Consider the partial Taylor polynomials p_m(x)=\\sum_{k=0}^{m}\\frac{x^k}{k!}. For n>m , p_n-p_m=\\sum_{k=m+1}^{n}\\frac{x^k}{k!}. Using the triangle inequality in the L^2 norm, \\|p_n-p_m\\|_{L^2} \\le \\sum_{k=m+1}^{n} \\left\\|\\frac{x^k}{k!}\\right\\|_{L^2}. But \\left\\|\\frac{x^k}{k!}\\right\\|_{L^2} = \\frac{1}{k!} \\left(\\int_0^1 x^{2k}\\,dx\\right)^{1/2} = \\frac{1}{k!\\sqrt{2k+1}}. The series \\sum_{k=0}^{\\infty}\\frac{1}{k!\\sqrt{2k+1}} converges. Hence (p_m) is Cauchy in the L^2 norm.`;

describe("prepareChunkMarkdown", () => {
  test("keeps subscripts on left-right norms", () => {
    const md = prepareChunkMarkdown(HILBERT_CHUNK);
    expect(md).toContain("\\right\\|_{L^2}$");
    expect(md).not.toContain("$$$");
    expect(md).toContain("The series");
    expect(md).toContain("L^2 norm");
    expect(md).not.toContain("$Hence");
  });

  test("keeps superscripts on left-right integrals", () => {
    const md = prepareChunkMarkdown(HILBERT_CHUNK);
    expect(md).toContain("\\right)^{1/2}$");
  });

  test("wraps variable assignments like p_m(x)=...", () => {
    const md = prepareChunkMarkdown(HILBERT_CHUNK);
    expect(md).toContain("$p_m(x)=\\sum_{k=0}^{m}\\frac{x^k}{k!}$");
    expect(md).toContain("$p_n-p_m=\\sum_{k=m+1}^{n}\\frac{x^k}{k!}$");
  });
});
