// TypeScript mirrors of the backend Pydantic contracts (common/models.py).
// Kept in lock-step with the API so a shape change surfaces as a type error, not a runtime bug.

export const ANOMALY_CLASSES = [
  "normal",
  "credential_misuse",
  "lateral_movement",
  "brute_force",
  "impossible_travel",
  "credential_stuffing",
  "device_spoofing",
  "low_and_slow_exfil",
  "insider_drift",
] as const;

export type AnomalyType = (typeof ANOMALY_CLASSES)[number];
export type EntityType = "user" | "service_account" | "edge_device";
export type AnalystVerdict = "confirmed" | "false_positive";

export interface DetectionScores {
  baseline: number;
  sequence: number;
  classifier_confidence: number;
  fused_raw: number | null;
}

export interface FeatureAttribution {
  feature: string;
  value: unknown;
  contribution: number;
  direction: "increases_risk" | "decreases_risk" | "neutral";
  baseline_value: unknown;
  description: string | null;
}

export interface CounterfactualChange {
  feature: string;
  actual: unknown;
  suggested: unknown;
  description: string | null;
}

export interface Counterfactual {
  changes: CounterfactualChange[];
  resulting_risk: number | null;
  original_risk: number | null;
  found: boolean;
  summary: string | null;
}

export interface MitreTechnique {
  technique_id: string;
  name: string;
  tactic: string | null;
  url: string | null;
  confidence: number;
}

export interface SequenceStepAttribution {
  position: number;
  token: string;
  score: number;
}

export interface BaselineComparison {
  fields: Record<string, { observed: unknown; typical: unknown; deviates: boolean }>;
  summary: string | null;
}

export interface Explanation {
  top_features: FeatureAttribution[];
  counterfactual: Counterfactual | null;
  sequence_attribution: SequenceStepAttribution[];
  mitre: MitreTechnique[];
  baseline_comparison: BaselineComparison | null;
  narrative: string | null;
  narrative_source: string;
}

export interface Detection {
  detection_id: string;
  entity_id: string;
  entity_type: EntityType;
  timestamp: string;
  event_ref: string | null;
  session_id: string | null;
  scores: DetectionScores;
  risk_score: number;
  risk_uncertainty: number;
  in_alert_budget: boolean;
  is_anomaly: boolean;
  anomaly_type: AnomalyType;
  anomaly_type_probs: Record<string, number>;
  detector_hits: string[];
  explanation: Explanation;
  campaign_id: string | null;
  cold_start: boolean;
  drift_flag: boolean;
  status: string;
  ground_truth_label: AnomalyType | null;
  created_at: string;
}

export interface CampaignStage {
  anomaly_type: AnomalyType;
  detection_id: string;
  timestamp: string;
  risk_score: number;
}

export interface Campaign {
  campaign_id: string;
  entity_id: string;
  entity_type: EntityType | null;
  started_at: string;
  last_activity: string;
  stages: CampaignStage[];
  detection_ids: string[];
  kill_chain: string[];
  max_risk: number;
  status: "open" | "closed";
  created_at: string;
  updated_at: string;
}

export interface FeedbackAdjustment {
  scope: string;
  scope_id: string | null;
  adjustment: number;
  previous_value: number | null;
  new_value: number | null;
}

export interface Feedback {
  feedback_id: string;
  detection_id: string;
  entity_id: string;
  analyst_verdict: AnalystVerdict;
  note: string | null;
  applied: FeedbackAdjustment | null;
  created_at: string;
}

export interface DashboardSummary {
  n_detections: number;
  n_anomalies: number;
  n_campaigns: number;
  n_feedback: number;
  by_type: Record<string, number>;
}

export interface EntityHistory {
  entity_id: string;
  n_detections: number;
  detections: Detection[];
}

export interface ModelMetrics {
  pr_auc?: number | null;
  roc_auc?: number | null;
  recall_at_1pct_budget?: number | null;
  macro_f1?: number | null;
  calibration_ece?: number | null;
  precision_at_k?: number[];
  confusion_matrix?: number[][];
  class_order?: string[];
  per_class?: Record<string, Record<string, number>>;
  [key: string]: unknown;
}

export interface DriftState {
  entity_id: string;
  psi?: number;
  status?: string;
  samples_seen?: number;
  [key: string]: unknown;
}

export interface HealthStatus {
  status: string;
  detail: string | null;
  latency_ms: number | null;
}

export interface ServiceHealth {
  service: string;
  status: string;
  version: string;
  artifact_schema_version: string | null;
  artifacts_ready: boolean;
  dependencies: Record<string, HealthStatus>;
  checked_at: string;
}
