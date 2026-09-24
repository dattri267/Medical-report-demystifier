package com.demystifier.backend.service;

import com.demystifier.backend.entity.ReviewCase;
import com.demystifier.backend.repository.ReviewCaseRepository;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.UUID;

@Service
public class ReviewQueueService {

    private final ReviewCaseRepository repository;

    public ReviewQueueService(ReviewCaseRepository repository) {
        this.repository = repository;
    }

    public ReviewCase flagCase(String payloadJson, String reason, String priority) {
        ReviewCase reviewCase = new ReviewCase();
        reviewCase.setCaseId(UUID.randomUUID().toString());
        reviewCase.setPayloadJson(payloadJson);
        reviewCase.setReason(reason);
        reviewCase.setPriority(priority);
        return repository.save(reviewCase);
    }

    public List<ReviewCase> getPendingCases() {
        return repository.findByReviewedFalseOrderByCreatedAtDesc();
    }

    public void markReviewed(Long id) {
        repository.findById(id).ifPresent(c -> {
            c.setReviewed(true);
            repository.save(c);
        });
    }
}
