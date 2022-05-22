export type ClassificationResult = {
    label: string;
    results: { probabilities: Float32Array; match: boolean;}[];
}