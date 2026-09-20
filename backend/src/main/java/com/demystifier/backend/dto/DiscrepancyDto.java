package com.demystifier.backend.dto;

public class DiscrepancyDto {
    private boolean flagged;
    private String reason;

    public DiscrepancyDto() {
    }

    public DiscrepancyDto(boolean flagged, String reason) {
        this.flagged = flagged;
        this.reason = reason;
    }

    public boolean isFlagged() {
        return flagged;
    }

    public void setFlagged(boolean flagged) {
        this.flagged = flagged;
    }

    public String getReason() {
        return reason;
    }

    public void setReason(String reason) {
        this.reason = reason;
    }
}