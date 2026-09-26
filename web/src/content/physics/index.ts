// Physics content registry. Each topic lives in ./topics/<id>.ts and default-exports a PhysicsTopic.
// Topics are collected automatically; a missing topic falls back to a visible "pending" stub.
import type { PhysicsTopic, TopicId } from "./types";

export * from "./types";

export const TOPIC_ORDER: TopicId[] = [
  "overview", "electrostatics", "impact-ionization", "btbt-gidl", "channel", "bjt-transport",
  "charge-balance", "photo", "stochastic-events", "first-passage", "local-states", "sweep-mc",
  "circuit-element", "parameters", "design-map", "open-problems", "validation", "numerics",
];

const modules = import.meta.glob<{ default: PhysicsTopic }>("./topics/*.ts", { eager: true });

const loaded = new Map<string, PhysicsTopic>();
for (const [path, mod] of Object.entries(modules)) {
  const topic = mod.default;
  const file = path.replace(/^.*\//, "").replace(/\.ts$/, "");
  if (topic && topic.id === file) loaded.set(topic.id, topic);
}

const stub = (id: TopicId): PhysicsTopic => ({
  id,
  title: { en: id, ko: id },
  summary: { en: "Content pending.", ko: "내용 준비 중." },
  sections: [],
});

export const PHYSICS_TOPICS: Record<TopicId, PhysicsTopic> = Object.fromEntries(
  TOPIC_ORDER.map((id) => [id, loaded.get(id) ?? stub(id)]),
) as Record<TopicId, PhysicsTopic>;

/** All equations by id (for cross references such as "Eq. (eq-charge-balance)"). */
export const EQUATION_INDEX = new Map(
  TOPIC_ORDER.flatMap((id) =>
    PHYSICS_TOPICS[id].sections.flatMap((s) => (s.equations ?? []).map((e) => [e.id, { topic: id, eq: e }] as const)),
  ),
);
