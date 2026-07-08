import { defineCollection } from "astro:content";
import { z } from "astro/zod";
import { glob } from "astro/loaders";
import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import timezone from "dayjs/plugin/timezone";
import config from "@/config";

export const BLOG_PATH = "src/content/posts";

dayjs.extend(utc);
dayjs.extend(timezone);

const hasExplicitTimezone = (value: string) =>
  /(?:[zZ]|[+-]\d{2}:\d{2})$/.test(value);

const parsePostDate = (value: string | Date) => {
  if (value instanceof Date) return value;

  const parsed = hasExplicitTimezone(value)
    ? dayjs(value)
    : dayjs.tz(value, config.site.timezone);

  return parsed.toDate();
};

const datetimeField = z
  .union([z.date(), z.string()])
  .transform(value => parsePostDate(value));

const posts = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: `./${BLOG_PATH}` }),
  schema: ({ image }) =>
    z.object({
      author: z.string().default(config.site.author),
      pubDatetime: datetimeField,
      modDatetime: datetimeField.optional().nullable(),
      title: z.string(),
      featured: z.boolean().optional(),
      draft: z.boolean().optional(),
      tags: z.array(z.string()).default(["others"]),
      ogImage: image().or(z.string()).optional(),
      description: z.string(),
      canonicalURL: z.string().optional(),
      hideEditPost: z.boolean().optional(),
      timezone: z.string().optional(),
    }),
});

const pages = defineCollection({
  loader: glob({ pattern: "**/[^_]*.{md,mdx}", base: "./src/content/pages" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    ogImage: z.string().optional(),
    canonicalURL: z.string().optional(),
  }),
});

export const collections = { posts, pages };
