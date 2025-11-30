import React from 'react';
import styled, { keyframes } from 'styled-components';

const spin = keyframes`
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
`;

const pulse = keyframes`
  0% { transform: scale(1); opacity: 1; }
  50% { transform: scale(1.1); opacity: 0.7; }
  100% { transform: scale(1); opacity: 1; }
`;

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 80px 20px;
  text-align: center;
`;

const SpinnerContainer = styled.div`
  position: relative;
  width: 120px;
  height: 120px;
  margin-bottom: 30px;
`;

const Spinner = styled.div`
  width: 100%;
  height: 100%;
  border: 8px solid #f3f3f3;
  border-top: 8px solid #007bff;
  border-radius: 50%;
  animation: ${spin} 1s linear infinite;
`;

const SpinnerCenter = styled.div`
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  font-size: 2rem;
  animation: ${pulse} 2s ease-in-out infinite;
`;

const LoadingText = styled.h2`
  margin: 0 0 15px 0;
  font-size: 1.5rem;
  color: #333;
  animation: ${pulse} 2s ease-in-out infinite;
`;

const LoadingSubtext = styled.p`
  margin: 0;
  color: #666;
  font-size: 1rem;
  line-height: 1.5;
  max-width: 400px;
`;

const ProcessingSteps = styled.div`
  margin-top: 30px;
  padding: 20px;
  background: #f8f9fa;
  border-radius: 10px;
  max-width: 500px;
`;

const StepItem = styled.div`
  display: flex;
  align-items: center;
  margin-bottom: 10px;
  font-size: 0.9rem;
  color: #555;
`;

const StepIcon = styled.span`
  margin-right: 10px;
  font-size: 1.2rem;
`;

function LoadingSpinner() {
  return (
    <LoadingContainer>
      <SpinnerContainer>
        <Spinner />
        <SpinnerCenter>🤖</SpinnerCenter>
      </SpinnerContainer>
      
      <LoadingText>AI Processing Your Image...</LoadingText>
      <LoadingSubtext>
        Our CodeFormer model is analyzing and restoring your image.
        This may take a few seconds depending on image size and server load.
      </LoadingSubtext>
      
      <ProcessingSteps>
        <h3 style={{ marginTop: 0, color: '#333' }}>Processing Steps:</h3>
        <StepItem>
          <StepIcon>📤</StepIcon>
          Uploading and preprocessing image
        </StepItem>
        <StepItem>
          <StepIcon>🔧</StepIcon>
          Applying degradation simulations
        </StepItem>
        <StepItem>
          <StepIcon>🧠</StepIcon>
          Running AI restoration models
        </StepItem>
        <StepItem>
          <StepIcon>📊</StepIcon>
          Generating comparison results
        </StepItem>
      </ProcessingSteps>
    </LoadingContainer>
  );
}

export default LoadingSpinner;