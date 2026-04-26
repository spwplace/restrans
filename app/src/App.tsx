import { Routes, Route } from 'react-router'
import Layout from './components/Layout'
import Home from './pages/Home'
import Paper from './pages/Paper'
import Figures from './pages/Figures'

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/paper" element={<Paper />} />
        <Route path="/figures" element={<Figures />} />
      </Routes>
    </Layout>
  )
}
