package com.demystifier.backend.dto;

import java.util.Map;

public class VisionResultDto {
    private Map<String, Double> findings;
    private String topFinding;
    private double topFindingConfidence;
    private String gradcamOverlayB64;

    public Map<String, Double> getFindings() {
        return findings;
    }

    public void setFindings(Map<String, Double> findings) {
        this.findings = findings;
    }

    public String getTopFinding() {
        return topFinding;
    }

    public void setTopFinding(String topFinding) {
        this.topFinding = topFinding;
    }

    public double getTopFindingConfidence() {
        return topFindingConfidence;
    }

    public void setTopFindingConfidence(double topFindingConfidence) {
        this.topFindingConfidence = topFindingConfidence;
    }

    public String getGradcamOverlayB64() {
        return gradcamOverlayB64;
    }

    public void setGradcamOverlayB64(String gradcamOverlayB64) {
        this.gradcamOverlayB64 = gradcamOverlayB64;
    }
}