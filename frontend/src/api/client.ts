// Thin typed client over the read API. One axios instance, one function per endpoint, so pages
// never build URLs by hand and every response is typed.

import axios from "axios";
import type {
  Campaign,
  DashboardSummary,
  Detection,
  DriftState,
  EntityHistory,
  Feedback,
  ModelMetrics,
  ServiceHealth,
} from "./types";

export const http = axios.create({
  baseURL: "/api/v1",
  timeout: 15000,
});

export interface DetectionQuery {
  skip?: number;
  limit?: number;
  sort?: "risk" | "time";
  anomaly_type?: string;
  entity_type?: string;
  cold_start?: boolean;
  min_risk?: number;
}

export async function getDashboardSummary(): Promise<DashboardSummary> {
  const { data } = await http.get<DashboardSummary>("/dashboard/summary");
  return data;
}

export async function listDetections(query: DetectionQuery = {}): Promise<Detection[]> {
  const { data } = await http.get<Detection[]>("/detections", { params: query });
  return data;
}

export async function getDetection(detectionId: string): Promise<Detection> {
  const { data } = await http.get<Detection>(`/detections/${detectionId}`);
  return data;
}

export async function getEntity(entityId: string, limit = 100): Promise<EntityHistory> {
  const { data } = await http.get<EntityHistory>(`/entities/${entityId}`, { params: { limit } });
  return data;
}

export async function listCampaigns(entityId?: string, limit = 50): Promise<Campaign[]> {
  const { data } = await http.get<Campaign[]>("/campaigns", {
    params: { entity_id: entityId, limit },
  });
  return data;
}

export async function getCampaign(campaignId: string): Promise<Campaign> {
  const { data } = await http.get<Campaign>(`/campaigns/${campaignId}`);
  return data;
}

export async function getMetrics(): Promise<ModelMetrics> {
  const { data } = await http.get<ModelMetrics>("/metrics");
  return data;
}

export async function getDrift(limit = 100): Promise<DriftState[]> {
  const { data } = await http.get<DriftState[]>("/drift", { params: { limit } });
  return data;
}

export async function getSystemHealth(): Promise<ServiceHealth> {
  const { data } = await http.get<ServiceHealth>("/system/health");
  return data;
}

export async function postFeedback(
  detectionId: string,
  verdict: "confirmed" | "false_positive",
  note?: string,
): Promise<Feedback> {
  const { data } = await http.post<Feedback>("/feedback", null, {
    params: { detection_id: detectionId, verdict, note },
  });
  return data;
}
