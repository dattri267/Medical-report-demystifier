package com.demystifier.backend.controller;

import com.demystifier.backend.entity.ReviewCase;
import com.demystifier.backend.service.ReviewQueueService;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/review-queue")
public class ReviewQueueController {

    private final ReviewQueueService reviewQueueService;

    public ReviewQueueController(ReviewQueueService reviewQueueService) {
        this.reviewQueueService = reviewQueueService;
    }

    @GetMapping
    public List<ReviewCase> getPending() {
        return reviewQueueService.getPendingCases();
    }

    @PostMapping("/{id}/mark-reviewed")
    public void markReviewed(@PathVariable Long id) {
        reviewQueueService.markReviewed(id);
    }
}