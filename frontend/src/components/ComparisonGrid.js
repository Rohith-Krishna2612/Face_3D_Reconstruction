import React, { useState } from 'react';
import styled from 'styled-components';

const GridContainer = styled.div`
  margin: 30px 0;
`;

const OriginalSection = styled.div`
  text-align: center;
  margin-bottom: 40px;
  padding: 25px;
  background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
  border-radius: 15px;
  box-shadow: 0 5px 15px rgba(0,0,0,0.1);
`;

const OriginalImage = styled.img`
  max-width: 400px;
  width: 100%;
  height: auto;
  border-radius: 10px;
  box-shadow: 0 8px 25px rgba(0,0,0,0.2);
  transition: transform 0.3s ease;
  
  &:hover {
    transform: scale(1.05);
  }
`;

const SectionTitle = styled.h2`
  margin-bottom: 20px;
  color: #333;
  font-size: 1.8rem;
  text-align: center;
`;

const ComparisonGrid = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
  gap: 25px;
  margin: 30px 0;
`;

const ComparisonCard = styled.div`
  background: white;
  border-radius: 15px;
  padding: 20px;
  box-shadow: 0 8px 25px rgba(0,0,0,0.1);
  transition: all 0.3s ease;
  margin-bottom: 30px;
  
  &:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 35px rgba(0,0,0,0.15);
  }
`;

const CardTitle = styled.h3`
  margin: 0 0 20px 0;
  color: #333;
  font-size: 1.5rem;
  text-transform: capitalize;
  text-align: center;
  padding-bottom: 15px;
  border-bottom: 3px solid #007bff;
`;

const ImageTriple = styled.div`
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 15px;
  margin-bottom: 20px;
  
  @media (max-width: 768px) {
    grid-template-columns: 1fr;
  }
`;

const ImageContainer = styled.div`
  position: relative;
  overflow: hidden;
  border-radius: 10px;
  cursor: pointer;
`;

const ComparisonImage = styled.img`
  width: 100%;
  height: 200px;
  object-fit: cover;
  border-radius: 10px;
  transition: transform 0.3s ease;
  
  &:hover {
    transform: scale(1.1);
  }
`;

const ImageLabel = styled.div`
  position: absolute;
  bottom: 8px;
  left: 8px;
  background: rgba(0,0,0,0.8);
  color: white;
  padding: 5px 12px;
  border-radius: 20px;
  font-size: 0.85rem;
  font-weight: 500;
`;

const QualityIndicator = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 15px;
  background: #f8f9fa;
  border-radius: 8px;
  margin-top: 15px;
  font-size: 0.9rem;
`;

const QualityBadge = styled.span`
  background: ${props => {
    if (props.type === 'degraded') return '#dc3545';
    if (props.type === 'restored') return '#28a745';
    if (props.type === 'original') return '#007bff';
    return '#6c757d';
  }};
  color: white;
  padding: 3px 8px;
  border-radius: 12px;
  font-size: 0.8rem;
`;

const ResetButton = styled.button`
  background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
  color: white;
  border: none;
  padding: 15px 40px;
  border-radius: 30px;
  font-size: 1.1rem;
  cursor: pointer;
  transition: all 0.3s ease;
  margin: 30px auto;
  display: block;
  box-shadow: 0 5px 15px rgba(0,123,255,0.3);
  
  &:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(0,123,255,0.4);
  }
`;

const ModalOverlay = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0,0,0,0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
  padding: 20px;
`;

const ModalContent = styled.div`
  background: white;
  border-radius: 15px;
  padding: 20px;
  max-width: 90vw;
  max-height: 90vh;
  overflow: auto;
`;

const ModalImage = styled.img`
  max-width: 100%;
  max-height: 70vh;
  object-fit: contain;
  border-radius: 10px;
`;

const CloseButton = styled.button`
  background: #dc3545;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 5px;
  cursor: pointer;
  margin-top: 15px;
  float: right;
`;

function ComparisonGrid({ results, onReset }) {
  const [modalImage, setModalImage] = useState(null);
  const [modalTitle, setModalTitle] = useState('');

  const openModal = (imageSrc, title) => {
    setModalImage(imageSrc);
    setModalTitle(title);
  };

  const closeModal = () => {
    setModalImage(null);
    setModalTitle('');
  };

  const degradationDisplayNames = {
    'blur': 'Gaussian Blur',
    'gaussian_noise': 'Gaussian Noise',
    'jpeg_compression': 'JPEG Compression',
    'downsampling': 'Downsampling'
  };

  const degradationEmojis = {
    'blur': '🌫️',
    'gaussian_noise': '🔇',
    'jpeg_compression': '📱',
    'downsampling': '🔍'
  };

  return (
    <GridContainer>
      <OriginalSection>
        <SectionTitle>📸 Original Image</SectionTitle>
        <OriginalImage 
          src={results.original} 
          alt="Original" 
          onClick={() => openModal(results.original, 'Original Image')}
        />
      </OriginalSection>

      <SectionTitle>🔄 Degradation & Restoration Comparisons</SectionTitle>
      <div style={{ marginBottom: '20px', textAlign: 'center', color: '#666', fontSize: '0.95rem' }}>
        Each row shows: <strong>Original Restored</strong> → <strong>Degraded Version</strong> → <strong>Restored from Degraded</strong>
      </div>
      
      {Object.entries(results.results).map(([degradationType, data]) => (
        <ComparisonCard key={degradationType}>
          <CardTitle>
            {degradationEmojis[degradationType]} {degradationDisplayNames[degradationType] || degradationType}
          </CardTitle>
          
          <ImageTriple>
            <ImageContainer onClick={() => openModal(results.original, 'Original Restored Image')}>
              <ComparisonImage 
                src={results.original} 
                alt="Original restored" 
              />
              <ImageLabel>Original Restored</ImageLabel>
            </ImageContainer>
            
            <ImageContainer onClick={() => openModal(data.degraded, `${degradationType} - Degraded`)}>
              <ComparisonImage 
                src={data.degraded} 
                alt={`${degradationType} degraded`} 
              />
              <ImageLabel>Degraded</ImageLabel>
            </ImageContainer>
            
            <ImageContainer onClick={() => openModal(data.restored, `${degradationType} - Restored`)}>
              <ComparisonImage 
                src={data.restored} 
                alt={`${degradationType} restored`} 
              />
              <ImageLabel>Restored from Degraded</ImageLabel>
            </ImageContainer>
          </ImageTriple>
          
          <QualityIndicator>
            <div>
              <QualityBadge type="original">Clean Input</QualityBadge>
              <span style={{ margin: '0 10px' }}>→</span>
              <QualityBadge type="degraded">Degraded</QualityBadge>
              <span style={{ margin: '0 10px' }}>→</span>
              <QualityBadge type="restored">AI Restored</QualityBadge>
            </div>
            <div style={{ fontSize: '0.8rem', color: '#666' }}>
              Click to zoom
            </div>
          </QualityIndicator>
        </ComparisonCard>
      ))}

      <ResetButton onClick={onReset}>
        🔄 Process New Image
      </ResetButton>

      {modalImage && (
        <ModalOverlay onClick={closeModal}>
          <ModalContent onClick={e => e.stopPropagation()}>
            <h3>{modalTitle}</h3>
            <ModalImage src={modalImage} alt={modalTitle} />
            <CloseButton onClick={closeModal}>Close</CloseButton>
          </ModalContent>
        </ModalOverlay>
      )}
    </GridContainer>
  );
}

export default ComparisonGrid;