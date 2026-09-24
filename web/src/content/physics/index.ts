// PLACEHOLDER — replaced by the physics-content work package.
// The UI must only rely on the exported names below.
import type { PhysicsTopic, TopicId } from "./types";

export * from "./types";

const stub = (id: TopicId, en: string, ko: string): PhysicsTopic => ({
  id,
  title: { en, ko },
  summary: { en: "Content pending.", ko: "내용 준비 중." },
  sections: [
    {
      heading: { en: "Placeholder", ko: "자리표시" },
      equations: [{ id: `eq-${id}-stub`, tex: "\\frac{dQ_B}{dt} = F(u, r)" }],
    },
  ],
});

export const TOPIC_ORDER: TopicId[] = [
  "overview", "electrostatics", "impact-ionization", "btbt-gidl", "channel", "bjt-transport",
  "charge-balance", "photo", "stochastic-events", "first-passage", "local-states", "sweep-mc",
  "circuit-element", "parameters", "design-map", "open-problems", "validation", "numerics",
];

export const PHYSICS_TOPICS: Record<TopicId, PhysicsTopic> = Object.fromEntries(
  TOPIC_ORDER.map((id) => [id, stub(id, id, id)]),
) as Record<TopicId, PhysicsTopic>;
