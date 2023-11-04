import { AppBar, Box, Paper } from '@mui/material'
import Container from '@mui/material/Container'
import MenuTabs from '@/components/MenuTabs'

export default function FloatingNav () {
  return (
    <AppBar sx={{
      backgroundColor: 'transparent',
      border: '0',
      boxShadow: '0'
    }}
    >
      <Container
        sx={{
          maxWidth: 'lg',
          position: 'relative',
          top: 25,
          display: 'flex',
          flexWrap: 'wrap'
        }}
      >
        <Box
          component='img'
          sx={{
            height: 66,
            width: 198,
            zIndex: 1101,
            display: 'inline-flex',
            position: 'relative',
            left: 0,
            top: -15
          }}
          alt='Monkey Head'
          src='/wrecksysLogo.svg'
        />
        <Paper
          elevation={20}
          sx={{
            bgColor: '#FFFFFF',
            borderRadius: 2,
            px: 0,
            mx: 0,
            zIndex: 1100,
            position: 'absolute',
            left: 0,
            width: '100%'
          }}
        >
          <MenuTabs />
        </Paper>
      </Container>
    </AppBar>
  )
}
