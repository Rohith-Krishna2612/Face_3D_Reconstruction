import React from 'react';
import styled from 'styled-components';

const HeaderContainer = styled.header`
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 30px 40px;
  text-align: center;
`;

const Title = styled.h1`
  margin: 0;
  font-size: 2.5rem;
  font-weight: 700;
  text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
`;

const Subtitle = styled.p`
  margin: 10px 0 0 0;
  font-size: 1.1rem;
  opacity: 0.9;
  font-weight: 300;
`;

const Description = styled.div`
  margin-top: 20px;
  padding: 20px;
  background: rgba(255,255,255,0.1);
  border-radius: 10px;
  backdrop-filter: blur(10px);
`;

const FeatureList = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-top: 15px;
`;

const Feature = styled.div`
  padding: 10px;
  background: rgba(255,255,255,0.1);
  border-radius: 8px;
  font-size: 0.9rem;
`;

function Header() {
  return (
    <HeaderContainer>
      <Title>🎭 Face 3D Reconstruction</Title>
      <Subtitle>AI-Powered Face Restoration & Enhancement</Subtitle>
      
      <Description>
        <p>
          Upload any face image and see it restored from multiple types of degradations.
          Our AI model handles blur, noise, compression, and downsampling artifacts.
        </p>
        
        <FeatureList>
          <Feature>🌫️ Blur Removal</Feature>
          <Feature>🔇 Noise Reduction</Feature>
          <Feature>📱 JPEG Enhancement</Feature>
          <Feature>🔍 Super Resolution</Feature>
        </FeatureList>
      </Description>
    </HeaderContainer>
  );
}

export default Header;