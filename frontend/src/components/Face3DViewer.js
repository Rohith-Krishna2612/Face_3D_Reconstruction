import React, { Suspense, useMemo } from 'react';
import styled from 'styled-components';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Html } from '@react-three/drei';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader';
import { useLoader } from '@react-three/fiber';
import * as THREE from 'three';

const ViewerSection = styled.div`
  margin: 20px 0 40px;
  padding: 24px;
  border-radius: 16px;
  background: linear-gradient(160deg, #f1f4ff 0%, #eef9ff 100%);
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
`;

const ViewerTitle = styled.h2`
  margin: 0 0 16px;
  color: #20304a;
  text-align: center;
`;

const ViewerShell = styled.div`
  height: 520px;
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid #d9e4f5;
  background: radial-gradient(circle at 20% 20%, #ffffff 0%, #dbe8fb 100%);

  @media (max-width: 768px) {
    height: 360px;
  }
`;

const Hint = styled.p`
  margin: 12px 0 0;
  text-align: center;
  color: #4f607e;
  font-size: 0.95rem;
`;

const Note = styled.div`
  margin-top: 12px;
  text-align: center;
  color: #5e6f8d;
  font-size: 0.9rem;
`;

function LoadingText() {
  return (
    <Html center>
      <div style={{ color: '#1d2f52', fontWeight: 600 }}>Loading 3D Mesh...</div>
    </Html>
  );
}

function FaceMesh({ url }) {
  const obj = useLoader(OBJLoader, url);

  const centered = useMemo(() => {
    const group = obj.clone();
    const box = new THREE.Box3().setFromObject(group);
    const center = new THREE.Vector3();
    box.getCenter(center);
    group.position.sub(center);

    const size = new THREE.Vector3();
    box.getSize(size);
    const maxAxis = Math.max(size.x, size.y, size.z) || 1;
    const scale = 2.2 / maxAxis;
    group.scale.setScalar(scale);

    group.traverse((child) => {
      if (child.isMesh) {
        child.material = new THREE.MeshStandardMaterial({
          color: '#e7b79f',
          roughness: 0.65,
          metalness: 0.05,
          flatShading: false,
        });
      }
    });

    return group;
  }, [obj]);

  return <primitive object={centered} />;
}

function Face3DViewer({ deca3d }) {
  const meshUrl = deca3d?.mesh_url;

  if (!meshUrl) {
    return null;
  }

  return (
    <ViewerSection>
      <ViewerTitle>Interactive 3D Face Output</ViewerTitle>
      <ViewerShell>
        <Canvas camera={{ position: [0, 0, 3.4], fov: 50 }}>
          <ambientLight intensity={0.75} />
          <directionalLight position={[2, 2, 2]} intensity={0.9} />
          <directionalLight position={[-2, -1, -2]} intensity={0.4} />
          <Suspense fallback={<LoadingText />}>
            <FaceMesh url={meshUrl} />
          </Suspense>
          <OrbitControls enableDamping dampingFactor={0.08} minDistance={1.2} maxDistance={8} />
        </Canvas>
      </ViewerShell>
      <Hint>Drag to rotate. Scroll to zoom. Right-drag to pan.</Hint>
      {deca3d?.note && <Note>{deca3d.note}</Note>}
    </ViewerSection>
  );
}

export default Face3DViewer;
