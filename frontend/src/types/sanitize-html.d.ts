declare module "sanitize-html" {
  interface IOptions {
    allowedTags?: string[];
    allowedAttributes?: Record<string, string[]>;
  }

  interface Defaults {
    allowedTags: string[];
    allowedAttributes: Record<string, string[]>;
  }

  interface SanitizeHtml {
    (dirty: string, options?: IOptions): string;
    defaults: Defaults;
  }

  const sanitizeHtml: SanitizeHtml;
  export default sanitizeHtml;
}
